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
Supported-CL-only AlphaGenome encoder for scAlphaGenome.

Runs AlphaGenome with per-cell-type CL conditioning, but ONLY for cell types
that are natively supported by AlphaGenome (listed in
data/metadata/alphagenome_supported_cell_types.txt). Unsupported cell types are
dropped entirely — no UBERON fallback.

Design rationale vs alphagenome_encoder_model.py:
  - That script uses organ-level UBERON for ALL cell types, producing one
    genomic-context embedding per gene (cell-type-agnostic).
  - This script uses the native CL term for each supported cell type, producing
    one track per (cell_type, gene). Cell types not in the supported list are
    discarded so the output is free of any fallback signal.

AlphaGenome RNA-seq native resolution is 1 bp (1,048,576 values per 1 MB window).
Mean-pooled to a configurable bin count (default 8192 = 128 bp/bin).

Output: data/ag/{organ}_{assay}_base_ag.npz
    features        float32 [n_supported_ct, n_genes, bins]  log1p RNA-seq track
    labels          float32 [n_supported_ct, n_genes]        pseudobulk log-CPM
    gene_ids        str     [n_genes]
    cell_type_ids   str     [n_supported_ct]                 supported CL terms
    resolution_bp   int     bp per bin = 1,048,576 / bins

Usage:
    uv run python3 src/alphagenome_encoder_base.py --organ lung --assay 10X --workers 8 --max_genes 50000
'''

INPUT_DIR  = 'data/processed'
OUTPUT_DIR = 'data/agcell'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run AlphaGenome with CL conditioning for supported cell types only.'
    )
    parser.add_argument('--organ',     required=True,
                        help='Organ name matching alphagenome_organ_map.json (e.g. "heart", "lung")')
    parser.add_argument('--assay',     required=True,
                        help='Assay name matching processed parquet filename (e.g. "10X")')
    parser.add_argument('--bins',      type=int, default=8192,
                        help='Number of bins to mean-pool the 1 MB track into. '
                             'Default 8192 = 128 bp/bin. Must divide 1,048,576 evenly.')
    parser.add_argument('--max_genes', type=int, default=None,
                        help='Limit to first N genes (for testing)')
    parser.add_argument('--workers',   type=int, default=4,
                        help='Concurrent API request threads (AlphaGenome is a remote API)')
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='Save a checkpoint every N completed tasks so the job is resumable')
    return parser.parse_args()


def construct_interval(gene) -> genome.Interval:
    return genome.Interval(
        chromosome=gene.chromosome,
        start=gene.start,
        end=gene.end,
        strand=gene.strand,
        name=gene.gene_id,
    )


def filter_rna_seq_output(rna_seq, interval):
    """Keep only the relevant strand and assay from the AlphaGenome RNA-seq output."""
    if rna_seq.num_tracks == 1:
        return rna_seq
    if interval.strand == '-':
        rna_seq = rna_seq.filter_to_nonpositive_strand()
    else:
        rna_seq = rna_seq.filter_to_nonnegative_strand()
    if rna_seq.num_tracks > 1:
        rna_seq = rna_seq.filter_tracks(
            [row['Assay title'] == 'polyA plus RNA-seq' for _, row in rna_seq.metadata.iterrows()]
        )
    return rna_seq


def extract_track(rna_seq, n_bins: int) -> np.ndarray:
    """Mean-pool the 1bp RNA-seq track to n_bins and apply log1p.

    Native output: (1,048,576, 1) at 1 bp resolution.
    Returns float32 array of shape (n_bins,).
    """
    values = rna_seq.values  # (n_positions, n_tracks)
    track = (values[:, 0] if values.ndim == 2 else values.flatten()).astype(np.float32)

    native_len = dna_client.SEQUENCE_LENGTH_1MB  # 1,048,576
    if len(track) != native_len:
        print(f'  [warn] unexpected track length {len(track)}, expected {native_len}')

    # Truncate to exactly native_len so reshape is safe, then mean-pool
    factor = native_len // n_bins
    track = track[:native_len].reshape(n_bins, factor).mean(axis=1)
    return np.log1p(track)


def main():
    print('Initializing AlphaGenome client...')
    model = dna_client.create(api_key=os.getenv('ALPHAGENOME_API_KEY'))

    args = parse_args()
    n_bins = args.bins

    native_len = dna_client.SEQUENCE_LENGTH_1MB
    if native_len % n_bins != 0:
        raise ValueError(f'--bins {n_bins} does not evenly divide {native_len}')
    resolution_bp = native_len // n_bins

    with open('data/metadata/alphagenome_supported_cell_types.txt') as f:
        supported_cl_terms = set(f.read().splitlines())

    print('Loading input data...')
    df = pd.read_parquet(f'{INPUT_DIR}/{args.organ}_{args.assay}_processed.parquet')
    if args.max_genes is not None:
        df = df.iloc[:args.max_genes]
    df = df.reset_index(drop=True)

    all_cell_types = [col for col in df.columns if col.startswith('CL:')]
    cell_types = [ct for ct in all_cell_types if ct in supported_cl_terms]
    dropped = len(all_cell_types) - len(cell_types)

    gene_ids = df['gene_id'].tolist()
    n_genes  = len(df)
    n_ct     = len(cell_types)

    if n_ct == 0:
        raise ValueError(
            f"No supported cell types found for organ '{args.organ}'. "
            f"Check that CL: columns exist in the parquet and appear in "
            f"alphagenome_supported_cell_types.txt."
        )

    print(f'Cell types: {len(all_cell_types)} total, {n_ct} supported, {dropped} dropped')
    print(f'Genes: {n_genes:,} | Bins: {n_bins} ({resolution_bp} bp/bin) | Workers: {args.workers}')
    est_mb = n_ct * n_genes * n_bins * 4 / 1e6
    print(f'Estimated feature size: {est_mb:.0f} MB (float32, before compression)')

    # Pre-allocate output arrays
    features = np.zeros((n_ct, n_genes, n_bins), dtype=np.float32)
    done     = np.zeros((n_ct, n_genes), dtype=bool)

    # Fill labels from the processed parquet — no API call needed
    labels = np.zeros((n_ct, n_genes), dtype=np.float32)
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}
    for gene_idx, row in df.iterrows():
        for ct in cell_types:
            labels[ct_idx[ct], gene_idx] = row[ct]

    # Resume from checkpoint if one exists from a previous interrupted run
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = f'{OUTPUT_DIR}/{args.organ}_{args.assay}_base_ag_checkpoint.npz'
    if os.path.exists(checkpoint_path):
        print(f'Resuming from checkpoint: {checkpoint_path}')
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        if ckpt['features'].shape == features.shape:
            features[:] = ckpt['features']
            done[:]     = ckpt['done']
            print(f'  {done.sum()} / {n_ct * n_genes} tasks already completed.')
        else:
            print(f'  [warn] checkpoint shape mismatch — starting fresh.')

    def fetch_track(ct_i, gene_idx, cl_term, interval):
        for attempt in range(6):
            try:
                output = model.predict_interval(
                    interval=interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
                    requested_outputs=[dna_client.OutputType.RNA_SEQ],
                    ontology_terms=[cl_term],
                )
                rna_seq = filter_rna_seq_output(output.rna_seq, interval)
                return ct_i, gene_idx, extract_track(rna_seq, n_bins)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f'Rate limit retries exceeded for ct_i={ct_i}, gene_idx={gene_idx}')

    pending = [
        (ct_i, ct, gene_idx, construct_interval(row))
        for ct_i, ct in enumerate(cell_types)
        for gene_idx, row in df.iterrows()
        if not done[ct_i, gene_idx]
    ]
    print(f'\nRunning {len(pending):,} API calls...')

    completed_since_ckpt = 0
    failed_tasks = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_track, ct_i, gene_idx, ct, interval): (ct_i, gene_idx)
            for ct_i, ct, gene_idx, interval in pending
        }
        for future in tqdm(as_completed(futures), total=len(pending)):
            ct_i, gene_idx = futures[future]
            try:
                _, _, track = future.result()
                features[ct_i, gene_idx] = track
                done[ct_i, gene_idx]     = True
                completed_since_ckpt += 1
            except Exception as e:
                print(f'\n  [error] ct={cell_types[ct_i]}, gene_idx={gene_idx} ({gene_ids[gene_idx]}): {e}')
                failed_tasks.append((ct_i, gene_idx))
                continue

            if completed_since_ckpt >= args.checkpoint_every:
                np.savez(checkpoint_path, features=features, done=done)
                completed_since_ckpt = 0

    if failed_tasks:
        print(f'\n[warn] {len(failed_tasks)} tasks failed: '
              f'{[(cell_types[ct_i], gene_ids[gi]) for ct_i, gi in failed_tasks[:10]]}'
              f'{"..." if len(failed_tasks) > 10 else ""}')

    print('\nSaving final output...')
    out_path = f'{OUTPUT_DIR}/{args.organ}_{args.assay}_base_ag.npz'
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        gene_ids=np.array(gene_ids, dtype=object),
        cell_type_ids=np.array(cell_types, dtype=object),
        resolution_bp=np.array(resolution_bp),
    )

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    actual_mb = os.path.getsize(out_path) / 1e6
    print(f'Done.')
    print(f'  features : {features.shape}  ({n_bins} bins at {resolution_bp} bp/bin)')
    print(f'  labels   : {labels.shape}')
    print(f'  output   : {out_path} ({actual_mb:.0f} MB compressed)')
    if failed_tasks:
        print(f'  failed   : {len(failed_tasks)} tasks skipped (listed above)')


if __name__ == '__main__':
    main()
