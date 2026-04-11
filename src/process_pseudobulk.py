import argparse
import pandas as pd

'''
process_pseudobulk.py
-------------------
Converts pseudobulk data files into a format that can be used for training the AlphaGenome model by performing the following steps:
1. Read in pseudobulk data files for a specified organ and assay method, retaining only the gene expression values and cell ontology IDs for each cell type
2. Map gene names in the pseudobulk data to genomic coordinates using the GTF annotation file provided by AlphaGenome (necessary for constructing Interval objects)

Arguments
---------
--organ         Name of organ the pseudobulk data file.
                Example: lung

--assay         Assay method of pseudobulk data file.
                Example: 10X

Output format
-------------
Parquet file where each row is a gene.
Columns: 
    gene_id: Ensembl gene ID (e.g., "ENSG00000000005.6")
    Cell type IDs: cell ontology ID (e.g., "CL:4028006"), with corresponding expression values
    Reference GTF columns: gene_name, chromosome, start, end, strand

Example usage
-------------
uv run python3 src/process_pseudobulk.py --organ lung --assay 10X
'''

# Global variables
INPUT_DIR = 'data/pseudobulk'
OUTPUT_DIR = 'data/processed'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", required=True, help="Organ name in input parquet file.")
    parser.add_argument("--assay", required=True, help="Assay name in input parquet file.")
    return parser.parse_args()

def process_expression(df):
    expr_df = df.set_index('cell_ontology_id') # Set cell ontology ID as row index
    expr_df.drop(columns=['cell_ontology_class', 'n_cells'], inplace=True) # Drop metadata columns
    
    print("Loading AlphaGenome's GTF annotation file...")
    gtf = pd.read_parquet('data/metadata/alphagenome_gtf.parquet')
    
    # Transpose pseudobulk data to have (genes x cell types) for merging with GTF genes
    expr_df = expr_df.T
    expr_df = expr_df.reset_index().rename(columns={'index': 'gene_id'})
    expr_df.columns.name = None

    # Merge the pseudobulk data with the GTF annotation to get genomic coordinates for each gene
    print("Merging pseudobulk data with gene annotation data...")
    expr_df_merged = expr_df.merge(gtf[[
        'gene_id', 'gene_name', 'Chromosome', 'Start', 'End', 'Strand'
    ]], on='gene_id', how='inner') # Drop genes in dataset that don't have available coordinates
    expr_df_merged.rename(columns={'Chromosome':'chromosome', 'Start':'start', 'End':'end', 'Strand':'strand'}, inplace=True)
    return expr_df_merged

def main():
    args = parse_args()
    
    # Read input pseudobulk data file
    print("Loading input data...")
    input_path = f'{INPUT_DIR}/{args.organ}_{args.assay}_pseudobulk.parquet'
    df = pd.read_parquet(input_path)
    df = df.reset_index()
    
    # Process pseudobulk data to prepare for AlphaGenome input
    print("Starting input data processing...")
    processed_df = process_expression(df)
    print("Finished processing.")

    # Save processed data to output file
    print("Saving output data...")
    output_path = f'{OUTPUT_DIR}/{args.organ}_{args.assay}_processed.parquet'
    processed_df.to_parquet(output_path)
    print("Complete!")

if __name__ == "__main__":
    main()