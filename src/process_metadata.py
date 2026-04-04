import argparse
from alphagenome.data import genome
from alphagenome.models import dna_client
import os
import json
import pandas as pd

'''
process_metadata.py
-------------------


Arguments
---------
--organs        Names of organs the data samples were from. (Used to identify the pseudobulk dataset file for cell type map construction.)
                Example: lung
                Example (multiple): lung heart

--assay         Assay method of data samples. (Used to identify the pseudobulk dataset file for cell type map construction.) 
                Example: 10X

Output format
-------------
This script generates the following metadata files stored in the 'data/metadata' directory:
1. 'alphagenome_organ_map.json': dictionary mapping organ name to UBERON ID based on the list of ontology terms in AlphaGenome
2. 'alphagenome_supported_cell_types.txt': list of CL IDs for cell types that are in the list of ontology terms in AlphaGenome
3. '<organ>_<assay>_cell_type_map.json': for each organ, dictionary mapping from CL ID to cell type name based on annotations in Tabula Sapiens

Example usage
-------------
uv run python3 src/process_metadata.py --organs lung heart --assay 10X
'''

# Global variables
METADATA_DIR = 'data/metadata'
DATASET_DIR = 'data/pseudobulk'
ORGAN_MAP = 'alphagenome_organ_map.json'
SUPPORTED_CELL_TYPES = 'alphagenome_supported_cell_types.txt'
CELL_TYPE_MAP = 'cell_type_map.json'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organs", nargs='+', help="Names of organs that each data sample is from.")
    parser.add_argument("--assay", required=True, help="Assay method of data samples (e.g., '10X').")
    return parser.parse_args()

def organ_map(model_metadata):
    # Map from anatomical part (organ/tissue) name to UBERON ID from the list of ontology terms supported by AlphaGenome
    organ_map = model_metadata[model_metadata['ontology_curie'].str.contains('^UBERON:', regex=True)].set_index('biosample_name')['ontology_curie'].to_dict()
    with open(f'{METADATA_DIR}/{ORGAN_MAP}', 'w') as f:
        json.dump(organ_map, f) # Saved as .json file

def supported_cell_types(model_metadata):
    # List of CL IDs that are in the list of ontology terms supported by AlphaGenome
    supported_cell_types = list(set([ont for ont in model_metadata['ontology_curie'] if ont.startswith('CL:')]))
    with open(f'{METADATA_DIR}/{SUPPORTED_CELL_TYPES}', 'w') as f:
        f.write('\n'.join(supported_cell_types)) # Saved as .txt file

def cell_type_map(organs, assay):
    # Cell ontology mapping from ID to type from given preprocessed scRNA-seq dataset
    for organ in organs:
        df = pd.read_parquet(f'{DATASET_DIR}/{organ}_{assay}_pseudobulk.parquet') # Read pseudobulk data (cell type x gene)
        df = df.reset_index()
        cell_type_map = df[['cell_ontology_class', 'cell_ontology_id']].copy().drop_duplicates()
        cell_type_map.set_index('cell_ontology_id', inplace=True)
        cell_type_map = cell_type_map['cell_ontology_class'].to_dict()
        with open(f'{METADATA_DIR}/{organ}_{assay}_{CELL_TYPE_MAP}', 'w') as f:
            json.dump(cell_type_map, f) # Saved as .json file

def main():
    args = parse_args()
    model = dna_client.create(api_key=os.getenv('ALPHAGENOME_API_KEY'))
    model_metadata = model.output_metadata().rna_seq

    print("Writing AlphaGenome organ mapping file ...")
    organ_map(model_metadata)
    print("Writing AlphaGenome supported cell types file ...")
    supported_cell_types(model_metadata)
    print("Writing cell type mapping files for pseudobulk datasets ...")
    cell_type_map(args.organs, args.assay)
    print("All metadata files successfully created.")

if __name__ == "__main__":
    main()