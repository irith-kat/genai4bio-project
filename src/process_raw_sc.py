"""
process_raw_sc.py
-----------------
Converts a raw Tabula Sapiens scRNA-seq .h5ad data file into a pseudobulk expression data frame saved as a Parquet file. Pseudobulk expression is 
computed by summing counts across all cells of the same type, then normalizing to counts per million (CPM) and log1p-transforming.

Arguments
---------
--input         Filename of the Tabula Sapiens .h5ad file (no directory).
                Example: Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad

--organ         Name of organ the data sample was obtained from.
                Example: lung

--assay         Value of the 'method' column to keep before aggregating. Filters to one sequencing technology so that count-scale differences 
                between platforms don't corrupt the pseudobulk. 
                Example: 10X

--gcs           Flag when set, uploads the output file to gs://<GCS_BUCKET>/pseudobulk/. If omitted, no upload is performed.
                Note: requires gcloud auth to be set.

--min-cells     Minimum number of cells a cell type must have to be included. Cell types below this threshold produce unreliable pseudobulk 
                estimates. 
                Default: 20.

Output format
-------------
Parquet file with shape [n_cell_types x n_genes].
Columns: 
    cell_ontology_class (e.g., "alveolar type 2 fibroblast cell")
    cell_ontology_id: CL ontology term (e.g., "CL:4028006")
    n_cells: number of cells that were aggregated for that cell type
    Ensembl gene IDs (e.g., "ENSG00000000005.6")
Values: log1p(CPM) pseudobulk expression

Example usage
-------------
uv run python3 src/process_raw_sc.py \\
    --input Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad \\
    --organ lung \\
    --assay 10X \\
    --min-cells 20
"""

import argparse
import os
import scipy.sparse
import numpy as np
import pandas as pd
import scanpy as sc
from google.cloud import storage

DATA_INPUT_DIR = "data/raw"
DATA_OUTPUT_DIR = "data/pseudobulk"
GCS_OUTPUT_DIR = "pseudobulk"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Filename of the input .h5ad file.")
    parser.add_argument("--organ", required=True, help="Name of organ data sample was obtained from.")
    parser.add_argument("--assay", required=True, help="Value of the 'method' column to filter for (e.g., '10X').")
    parser.add_argument("--gcs", action="store_true", help="Upload the output parquet to GCS bucket.")
    parser.add_argument("--min-cells", type=int, default=20, help="Minimum cells per cell type to include.")
    return parser.parse_args()

def load_and_filter(h5ad_path: str, assay: str | None) -> sc.AnnData:
    # Load the .h5ad file
    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Filter to specified assay
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

    project = os.getenv("GCP_PROJECT")
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
    output_fname = f'{args.organ}_{args.assay}_pseudobulk.parquet'
    input_path = os.path.join(DATA_INPUT_DIR, args.input)
    output_path = os.path.join(DATA_OUTPUT_DIR, output_fname)

    # Load the .h5ad file, filter to one assay if specified, filter to protein-coding genes, and compute pseudobulk expression
    adata = load_and_filter(input_path, args.assay)
    adata = filter_genes(adata)
    df = compute_pseudobulk(adata, args.min_cells)

    # Write the resulting pseudobulk DataFrame to a Parquet file
    print(f"Writing {df.shape[0]} cell types x {df.shape[1] - 2} genes to {output_path} ...")
    df.to_parquet(output_path)
    print("Done.")
    print(f"\nOutput shape: {df.shape}")
    print(f"Cell types: {df.index.tolist()[:5]} ...")

    # Optionally upload to GCS
    if args.gcs:
        blob_path = f"{GCS_OUTPUT_DIR}/{output_fname}"
        upload_to_gcs(output_path, blob_path)

if __name__ == "__main__":
    main()
