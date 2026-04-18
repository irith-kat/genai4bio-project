import argparse
import os
import pandas as pd
import scanpy as sc

'''
Extracts scVI latent embeddings from a Tabula Sapiens dataset and computes the
mean embedding per cell type. Output is intended to be paired with AlphaGenome
features from alphagenome_encoder.py for decoder model training.

Output: data/scvi/{organ}_{assay}_cell_type_embeds.parquet
    Index:  cell_ontology_id (CL: ID) — joins directly on cell_type_ids from alphagenome_encoder output
    Columns:
        cell_ontology_class     human-readable cell type name
        n_cells                 number of cells averaged over
        scvi_0 ... scvi_49      mean scVI latent dimensions

Example:
uv run python3 src/cell_type_encoder.py --input Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad --organ lung --assay 10X
'''

# Global variables
DATA_INPUT_DIR = 'data/raw'
DATA_OUTPUT_DIR = 'data/scvi'
OUTPUT_FILE_SUFFIX = 'cell_type_embeds.parquet'
SCVI_KEY = 'X_scvi'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Filename of the input .h5ad file.')
    parser.add_argument('--organ', required=True, help='Name of organ the data sample was obtained from.')
    parser.add_argument('--assay', required=True, help="Value of the 'method' column to filter for (e.g., '10X').")
    parser.add_argument('--min-cells', type=int, default=20, help='Minimum cells per cell type to include.')
    return parser.parse_args()

def load_and_filter(h5ad_path, assay):
    print(f'Loading {h5ad_path} ...')
    adata = sc.read_h5ad(h5ad_path)
    print(f'  Loaded {adata.n_obs:,} cells')

    available = adata.obs['method'].unique().tolist()
    if assay not in available:
        raise ValueError(f"Assay '{assay}' not found. Available values: {available}")
    adata = adata[adata.obs['method'] == assay].copy()
    print(f"  After filtering to assay='{assay}': {adata.n_obs:,} cells")
    return adata

def compute_mean_embeddings(adata, min_cells):
    if SCVI_KEY not in adata.obsm:
        raise KeyError(f"'{SCVI_KEY}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")

    embeddings = adata.obsm[SCVI_KEY]  # [n_cells, latent_dim]
    latent_dim = embeddings.shape[1]
    scvi_cols = [f'scvi_{i}' for i in range(latent_dim)]

    cell_types = adata.obs['cell_ontology_class']
    cell_ids = adata.obs['cell_ontology_id']

    counts_per_type = cell_types.value_counts()
    valid_types = counts_per_type[counts_per_type >= min_cells].index
    n_dropped = len(counts_per_type) - len(valid_types)
    if n_dropped > 0: print(f'  Dropping {n_dropped} cell types with fewer than {min_cells} cells')

    rows = []
    for ct in sorted(valid_types):
        mask = (cell_types == ct).values
        rows.append({
            'cell_ontology_class': ct,
            'cell_ontology_id': cell_ids[cell_types == ct].iloc[0],
            'n_cells': int(mask.sum()),
            **dict(zip(scvi_cols, embeddings[mask].mean(axis=0))),
        })

    print(f'  Computed mean embeddings for {len(rows)} cell types')
    return pd.DataFrame(rows).set_index('cell_ontology_id')

def main():
    args = parse_args()
    input_path = os.path.join(DATA_INPUT_DIR, args.input)
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(DATA_OUTPUT_DIR, f'{args.organ}_{args.assay}_{OUTPUT_FILE_SUFFIX}')

    adata = load_and_filter(input_path, args.assay)
    df = compute_mean_embeddings(adata, args.min_cells)

    print(f'Writing {df.shape[0]} cell types x {df.shape[1] - 2} embedding dims to {output_path} ...')
    df.to_parquet(output_path)
    print('Done.')
    print(f'\nOutput shape: {df.shape}')
    print(f'Cell types: {df.index.tolist()[:5]} ...')

if __name__ == '__main__':
    main()
