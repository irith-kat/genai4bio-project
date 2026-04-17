import argparse
import json
import os
import pandas as pd
import numpy as np
from alphagenome.data import genome
from alphagenome.models import dna_client
from tqdm import tqdm

'''
Example:
uv run python3 src/run_alphagenome_base.py --organ lung --assay 10X --max_genes 500
'''

# Global variables
INPUT_DIR = 'data/processed'
OUTPUT_DIR = 'output'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--organ', required=True, help='Organ name in input parquet file.')
    parser.add_argument('--assay', required=True, help='Assay name in input parquet file.')
    parser.add_argument('--max_genes', type=int, default=None, help='Maximum number of genes from dataset to process.')
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
    if rna_seq.num_tracks == 1: return rna_seq
    if interval.strand == '-': rna_seq = rna_seq.filter_to_nonpositive_strand()
    else: rna_seq = rna_seq.filter_to_nonnegative_strand()
    if rna_seq.num_tracks > 1: rna_seq = rna_seq.filter_tracks([row['Assay title'] == 'polyA plus RNA-seq' for _, row in rna_seq.metadata.iterrows()])
    return rna_seq

def main():
    model = dna_client.create(api_key=os.getenv('ALPHAGENOME_API_KEY'))
    with open('data/metadata/alphagenome_organ_map.json', 'r') as f:
        model_uberon_ontology_terms = json.load(f)
    with open('data/metadata/alphagenome_supported_cell_types.txt', 'r') as f:
        model_cl_ontology_terms = f.read().splitlines()
    args = parse_args()

    print('Loading input data...')
    input_path = f'{INPUT_DIR}/{args.organ}_{args.assay}_processed.parquet'
    df = pd.read_parquet(input_path)
    if args.max_genes is not None: df = df.iloc[:args.max_genes] # Subset to specified number of genes for faster runtime during testing
    results = []
    
    # Iterate over each CL ID
    cell_types = [col for col in df.columns if col.startswith('CL:')]
    print(f'Number of cell types: {len(cell_types)}')
    for i in range(len(cell_types)):
        cell_type = cell_types[i]
        print(f'Cell type {cell_type} ({i+1}/{len(cell_types)}):')
        # Select most granular ontology term available for the organ/cell type
        cl_ontology_used = False
        ontology_term = model_uberon_ontology_terms[args.organ]
        if cell_type in model_cl_ontology_terms:
            ontology_term = cell_type
            cl_ontology_used = True
            print(f'Conditioning Alpha Genome on cell ontology term {ontology_term}.')
        else:
            print(f'Cell type not supported. Conditioning Alpha Genome on organ-level ontology term {ontology_term}.')
        
        # Iterate through each gene ID
        diffs_sum, diffs_avg, diffs_max = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df)):

            # For each gene, predict RNA-seq signal tracks for the 1MB region centered around the gene
            interval = construct_interval(row)
            output = model.predict_interval(
                interval=interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
                requested_outputs=[dna_client.OutputType.RNA_SEQ],
                ontology_terms=[ontology_term],
            )
            rna_seq = filter_rna_seq_output(output.rna_seq, interval)

            # Extract predicted expression for the gene
            track_start = rna_seq.interval.start
            gene_start = interval.start - track_start
            gene_end = interval.end - track_start
            rna_seq_gene = rna_seq[gene_start:gene_end]

            # Expression across the gene body
            pred_expr_sum = rna_seq_gene.values.sum()
            pred_expr_avg = rna_seq_gene.values.mean()
            pred_expr_max = rna_seq_gene.values.max()

            # Compare predicted expression with ground truth
            gt_expr = row[cell_type]
            diff_sum = abs(pred_expr_sum - gt_expr)
            diff_avg = abs(pred_expr_avg - gt_expr)
            diff_max = abs(pred_expr_max - gt_expr)
            diffs_sum.append(diff_sum)
            diffs_avg.append(diff_avg)
            diffs_max.append(diff_max)
        
        # Average absolute difference for the cell type
        avg_diff_sum, std_diff_sum = np.mean(diffs_sum), np.std(diffs_sum)
        avg_diff_avg, std_diff_avg = np.mean(diffs_avg), np.std(diffs_avg)
        avg_diff_max, std_diff_max = np.mean(diffs_max), np.std(diffs_max)
        print(f'Average absolute difference in sum expression: {avg_diff_sum:.4f} (std: {std_diff_sum:.4f})')
        print(f'Average absolute difference in average expression: {avg_diff_avg:.4f} (std: {std_diff_avg:.4f})')
        print(f'Average absolute difference in max expression: {avg_diff_max:.4f} (std: {std_diff_max:.4f})')

        results.append({
            'cell_type': cell_type,
            'cl_ontology_used': cl_ontology_used,
            'avg_diff_sum': avg_diff_sum,
            'std_diff_sum': std_diff_sum,
            'avg_diff_avg': avg_diff_avg,
            'std_diff_avg': std_diff_avg,
            'avg_diff_max': avg_diff_max,
            'std_diff_max': std_diff_max,
        })
    
    # Save results to output file
    print('Saving output data...')
    pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}/{args.organ}_{args.assay}_base_res.csv', index=False)

if __name__ == "__main__":
    main()