"""
preprocess.py — Tabula Sapiens pseudobulk preprocessing

Converts a Tabula Sapiens .h5ad file into a pseudobulk expression matrix saved
as a Parquet file. Pseudobulk is computed by summing raw counts across all cells
of the same cell type, then normalizing to counts per million (CPM) and
log1p-transforming. This produces one expression vector per cell type, which is
our prediction target.

Raw counts are used for aggregation (not pre-normalized values) because summing
then normalizing is statistically correct; averaging pre-normalized values from
cells with very different library sizes is not.

Arguments
---------
--input     Filename of the Tabula Sapiens .h5ad file (no directory).
            Example: Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad

--output    Filename of the output .parquet file (no directory).
            Example: lung_pseudobulk.parquet

--input-dir Directory containing the input .h5ad file. Default: data/raw

--output-dir Directory to write the output .parquet file. Default: data/processed

--assay     Value of the `method` column to keep before aggregating. Filters to
            one sequencing technology so that count-scale differences between
            platforms don't corrupt the pseudobulk. Choices depend on what is
            present in the file; common values across Tabula Sapiens files are:
              "10X"       — 10x Chromium droplet sequencing (more cells, sparser)
              "smartseq"  — Smart-seq2 full-length sequencing (fewer cells, denser)
            If not provided, all assays are pooled (not recommended unless the
            file contains only one assay).

--gcs       Flag. When set, uploads the output parquet to
            gs://<GCS_BUCKET>/preprocessed/<output>. Bucket is read from
            GCS_BUCKET in the environment. If omitted, no upload is performed.
            Requires gcloud auth or GOOGLE_APPLICATION_CREDENTIALS to be set.

--min-cells Minimum number of cells a cell type must have to be included.
            Cell types below this threshold produce unreliable pseudobulk
            estimates. Default: 20.
            Example: --min-cells 30

Output format
-------------
Parquet file with shape [n_cell_types x n_genes].
  - Index: cell_ontology_class (string, e.g. "endothelial cell of artery")
  - Columns: Ensembl gene IDs (e.g. "ENSG00000141510")
  - Values: log1p(CPM) pseudobulk expression (float32)
  - Metadata columns prepended: cell_ontology_id, n_cells
      cell_ontology_id — CL ontology term (e.g. "CL:0000413"), for use as input
                         to the cell type encoder
      n_cells          — number of cells that were aggregated for that cell type,
                         useful for filtering or weighting during training

Gene set
--------
Filtered to protein-coding genes only, identified by Ensembl IDs starting with
"ENSG" and excluding mitochondrial (MT-) and ERCC spike-in genes. This reduces
~60k features to ~19k and matches the gene set AlphaGenome was trained on.

Example usage
-------------
    # Local only (directories default to data/raw and data/processed)
    uv run python3 preprocess.py \\
        --input  Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad \\
        --output lung_10x_pseudobulk.parquet \\
        --assay  10X \\
        --min-cells 20

    # Override directories
    uv run python3 preprocess.py \\
        --input      Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad \\
        --input-dir  /mnt/data/raw \\
        --output     lung_10x_pseudobulk.parquet \\
        --output-dir /mnt/data/processed \\
        --assay      10X

    # Local + upload to gs://<GCS_BUCKET>/preprocessed/lung_10x_pseudobulk.parquet
    uv run python3 preprocess.py \\
        --input  Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad \\
        --output lung_10x_pseudobulk.parquet \\
        --assay  10X \\
        --gcs
"""

import argparse
import os
import scipy.sparse
import numpy as np
import pandas as pd
import scanpy as sc
from google.cloud import storage

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pseudobulk expression from a Tabula Sapiens .h5ad file."
    )
    parser.add_argument("--input", required=True, help="Filename of the input .h5ad file (no directory).")
    parser.add_argument("--output", required=True, help="Filename of the output .parquet file (no directory).")
    parser.add_argument("--input-dir", default="data/raw", help="Directory containing the input file (default: data/raw).")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to write the output file (default: data/processed).")
    parser.add_argument(
        "--assay",
        default=None,
        help="Value of the 'method' column to filter for (e.g. '10X' or 'smartseq'). "
             "If omitted, all assays are pooled.",
    )
    parser.add_argument(
        "--gcs",
        action="store_true",
        help="Upload the output parquet to gs://<GCS_BUCKET>/preprocessed/<output>. "
             "Bucket is read from GCS_BUCKET in the environment.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=20,
        help="Minimum cells per cell type to include (default: 20).",
    )
    return parser.parse_args()

def load_and_filter(h5ad_path: str, assay: str | None) -> sc.AnnData:
    # Load the .h5ad file
    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Filter to one assay if specified (e.g. "10X" or "smartseq")
    if assay is not None:
        if "method" not in adata.obs.columns:
            raise ValueError("Column 'method' not found in .obs. Cannot filter by assay.")
        available = adata.obs["method"].unique().tolist()
        if assay not in available:
            raise ValueError(
                f"Assay '{assay}' not found. Available values: {available}"
            )
        adata = adata[adata.obs["method"] == assay].copy()
        print(f"  After filtering to assay='{assay}': {adata.n_obs:,} cells")

    return adata

def filter_genes(adata: sc.AnnData) -> sc.AnnData:
    # Keep protein-coding genes: Ensembl IDs starting with ENSG, excluding MT and ERCC
    var = adata.var
    ensembl_mask = var["ensembl_id"].str.startswith("ENSG", na=False)
    mt_mask = var["mt"] if "mt" in var.columns else pd.Series(False, index=var.index)
    ercc_mask = var["ercc"] if "ercc" in var.columns else pd.Series(False, index=var.index)

    keep = ensembl_mask & ~mt_mask & ~ercc_mask
    adata = adata[:, keep].copy()
    print(f"  After gene filtering: {adata.n_vars:,} protein-coding genes retained")
    return adata

def compute_pseudobulk(adata: sc.AnnData, min_cells: int) -> pd.DataFrame:
    # Sum raw_counts per cell_ontology_class, normalize to CPM, log1p-transform
    if "raw_counts" not in adata.layers:
        raise ValueError(
            "'raw_counts' layer not found."
        )

    cell_types = adata.obs["cell_ontology_class"]
    cell_ids = adata.obs["cell_ontology_id"]

    # Count cells per type and filter out cell types with too few cells
    counts_per_type = cell_types.value_counts()
    valid_types = counts_per_type[counts_per_type >= min_cells].index
    n_dropped = len(counts_per_type) - len(valid_types)
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} cell types with fewer than {min_cells} cells")

    ensembl_ids = adata.var["ensembl_id"].values
    rows = []

    # Process each cell type: sum raw counts, normalize to CPM, log1p-transform
    for ct in sorted(valid_types):
        mask = (cell_types == ct).values
        ct_adata = adata[mask]
        n_cells = mask.sum()

        # Sum raw counts across cells
        X = ct_adata.layers["raw_counts"]
        if scipy.sparse.issparse(X):
            raw_sum = np.asarray(X.sum(axis=0)).flatten()
        else:
            raw_sum = X.sum(axis=0)

        # Normalize to CPM
        total = raw_sum.sum()
        if total == 0:
            print(f"  Warning: cell type '{ct}' has zero total counts, skipping.")
            continue
        cpm = raw_sum / total * 1e6

        # Log1p-transform
        log_cpm = np.log1p(cpm).astype(np.float32)

        # Get a representative cell_ontology_id for this cell type (they should in theory all be the same)
        cl_id = cell_ids[cell_types == ct].iloc[0]

        # Append the pseudobulk expression vector for this cell type, along with metadata
        rows.append({
            "cell_ontology_class": ct,
            "cell_ontology_id": cl_id,
            "n_cells": int(n_cells),
            **dict(zip(ensembl_ids, log_cpm)),
        })

    # Returns a DataFrame [cell_type x gene] with metadata columns prepended
    print(f"  Computed pseudobulk for {len(rows)} cell types")
    df = pd.DataFrame(rows).set_index("cell_ontology_class")
    return df

def upload_to_gcs(local_path: str, blob_path: str) -> None:
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise ValueError("GCS_BUCKET must be set in .env to upload to GCS.")

    project = os.getenv("GCP_PROJECT") or None
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    gcs_uri = f"gs://{bucket_name}/{blob_path}"
    print(f"Uploading {local_path} to {gcs_uri} ...")
    blob.upload_from_filename(local_path)
    print(f"  Upload complete.")

def main():
    # Parse CLI arguments
    args = parse_args()
    input_path = os.path.join(args.input_dir, args.input)
    output_path = os.path.join(args.output_dir, args.output)

    # Load the .h5ad file, filter to one assay if specified, filter to protein-coding genes, and compute pseudobulk expression
    adata = load_and_filter(input_path, args.assay)
    adata = filter_genes(adata)
    df = compute_pseudobulk(adata, args.min_cells)

    # Write the resulting pseudobulk DataFrame to a Parquet file
    print(f"Writing {df.shape[0]} cell types x {df.shape[1] - 2} genes to {output_path} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_parquet(output_path)
    print("Done.")
    print(f"\nOutput shape: {df.shape}")
    print(f"Cell types: {df.index.tolist()[:5]} ...")

    # Optionally upload to GCS
    if args.gcs:
        blob_path = f"preprocessed/{args.output}"
        upload_to_gcs(output_path, blob_path)

if __name__ == "__main__":
    main()
