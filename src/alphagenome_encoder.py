import argparse
import json
import os
import time
import grpc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from alphagenome.data import genome
from alphagenome.models import dna_client
from tqdm import tqdm

'''
Runs the AlphaGenome API and saves RNA-seq feature vectors alongside ground
truth expression labels. Intended as input to a decoder that maps
(sequence_features, cell_type_embedding) -> scalar expression.

API calls are deduplicated: cell types sharing the same ontology term (e.g.,
unsupported cell types falling back to the organ-level UBERON term) share one
API call per gene. Calls are parallelized with --workers concurrent threads
(AlphaGenome is a remote API so speedup comes from concurrent requests, not GPU).

Features are saved at 2048 bp resolution (BINS=512). Each bin is the mean
of 2048 bp of the 1 Mbp window, log1p-transformed.

Output: output/{organ}_{assay}_alphagenome_features.npz
    features        float32 [n_cell_types, n_genes, BINS]  log1p mean-pooled
    labels          float32 [n_cell_types, n_genes]
    gene_ids        str     [n_genes]
    cell_type_ids   str     [n_cell_types]
    ontology_terms  str     [n_cell_types]

Examples:
uv run python3 src/alphagenome_encoder.py --organ lung --assay 10X --workers 8 --max_genes 10000
uv run python3 src/alphagenome_encoder.py --organ lung --assay 10X --workers 8
'''

BINS = 512  # 2^20 bp / 2048 bp per bin = 512 bins

# Global variables
INPUT_DIR = 'data/processed'
OUTPUT_DIR = 'data/ag'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--organ', required=True)
    parser.add_argument('--assay', required=True)
    parser.add_argument('--max_genes', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent API request threads.')
    return parser.parse_args()

def construct_interval(gene) -> genome.Interval:
    # Build a genomic interval object from a gene row in the processed parquet
    return genome.Interval(
        chromosome=gene.chromosome,
        start=gene.start,
        end=gene.end,
        strand=gene.strand,
        name=gene.gene_id,
    )

def filter_rna_seq_output(rna_seq, interval):
    # AlphaGenome returns tracks for both strands and different RNA-seq types; keep only one track
    if rna_seq.num_tracks == 1: return rna_seq
    if interval.strand == '-': rna_seq = rna_seq.filter_to_nonpositive_strand()
    else: rna_seq = rna_seq.filter_to_nonnegative_strand()
    if rna_seq.num_tracks > 1: rna_seq = rna_seq.filter_tracks([row['Assay title'] == 'polyA plus RNA-seq' for _, row in rna_seq.metadata.iterrows()])
    return rna_seq

def bin_track(values):
    # Mean-pool the full 1 Mbp track into 512 bins of 2048 bp each
    values = values.flatten().astype(np.float32)[:2**20]
    binned = values.reshape(BINS, 2**20 // BINS).mean(axis=1)
    return np.log1p(binned)  # Log-transform (helps with ML training)

def main():
    print("Initializing AlphaGenome client...")
    model = dna_client.create(api_key=os.getenv('ALPHAGENOME_API_KEY'))
    with open('data/metadata/alphagenome_organ_map.json', 'r') as f:
        model_uberon_ontology_terms = json.load(f)
    with open('data/metadata/alphagenome_supported_cell_types.txt', 'r') as f:
        model_cl_ontology_terms = f.read().splitlines()
    args = parse_args()

    print('Loading input data...')
    df = pd.read_parquet(f'{INPUT_DIR}/{args.organ}_{args.assay}_processed.parquet')
    if args.max_genes is not None: df = df.iloc[:args.max_genes]
    df = df.reset_index(drop=True)

    cell_types = [col for col in df.columns if col.startswith('CL:')]
    gene_ids = df['gene_id'].tolist()
    n_cell_types, n_genes, n_bins = len(cell_types), len(df), BINS

    # Map each cell type to its ontology term, falling back to organ-level if unsupported
    organ_term = model_uberon_ontology_terms[args.organ]
    ontology_terms = {ct: ct if ct in model_cl_ontology_terms else organ_term for ct in cell_types}

    # Group cell types by ontology term to deduplicate API calls
    term_to_cell_types = {}
    for ct, term in ontology_terms.items():
        term_to_cell_types.setdefault(term, []).append(ct)

    n_unique_terms = len(term_to_cell_types)
    n_total_calls = n_unique_terms * n_genes
    print(f'Cell types: {n_cell_types} | Unique ontology terms: {n_unique_terms} | Genes: {n_genes} | Raw bins: {n_bins}')
    print(f'Total API calls: {n_total_calls:,} (vs {n_cell_types * n_genes:,} without deduplication) | Workers: {args.workers}')

    # Pre-allocate output arrays
    features = np.zeros((n_cell_types, n_genes, n_bins), dtype=np.float32) # AlphaGenome binned tracks
    labels   = np.zeros((n_cell_types, n_genes), dtype=np.float32) # Ground truth pseudobulk expression
    ct_index = {ct: i for i, ct in enumerate(cell_types)}

    # Pre-fill labels (no API needed — read directly from the processed parquet)
    for gene_idx, row in df.iterrows():
        for ct in cell_types:
            labels[ct_index[ct], gene_idx] = row[ct]

    # One API task per unique (ontology_term, gene) pair; retries with exponential backoff on rate limit
    def fetch_binned_track(term, gene_idx, interval):
        for attempt in range(6):
            try:
                output = model.predict_interval(
                    interval=interval.resize(dna_client.SEQUENCE_LENGTH_1MB),  # center 1MB window on gene
                    requested_outputs=[dna_client.OutputType.RNA_SEQ],
                    ontology_terms=[term],
                )
                rna_seq = filter_rna_seq_output(output.rna_seq, interval)
                return term, gene_idx, bin_track(rna_seq.values)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16, 32 seconds
                else:
                    raise
        raise RuntimeError(f'Rate limit retries exceeded (gene_idx={gene_idx}, term={term})')

    # Build task list and submit all API calls concurrently
    tasks = [
        (term, gene_idx, construct_interval(row))
        for term in term_to_cell_types
        for gene_idx, row in df.iterrows()
    ]

    print('\nRunning API calls...')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_binned_track, term, gene_idx, interval): (term, gene_idx)
                   for term, gene_idx, interval in tasks}
        for future in tqdm(as_completed(futures), total=n_total_calls):
            term, gene_idx, binned = future.result()
            # Assign the same binned track to all cell types sharing this term
            for ct in term_to_cell_types[term]:
                features[ct_index[ct], gene_idx] = binned

    # Save features and labels
    print('\nSaving output data...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        f'{OUTPUT_DIR}/{args.organ}_{args.assay}_alphagenome_features.npz',
        features=features,
        labels=labels,
        gene_ids=np.array(gene_ids),
        cell_type_ids=np.array(cell_types),
        ontology_terms=np.array([ontology_terms[ct] for ct in cell_types]),
    )
    print(f'Done. features={features.shape}, labels={labels.shape}')

if __name__ == '__main__':
    main()
