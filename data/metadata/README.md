### Ontological Metadata

This data folder contains the various files produced from `process_metadata.py`: mapping dictionaries and lists for the ontology terms used both in the different Tabula Sapiens scRNA-seq datasets and within the AlphaGenome model/API.

There are three types of biological classes that are important in this project:
* **Organs**, denoted with prefix `UBERON:`
* **Cell types**, denoted with prefix `CL:`
* **Genes**, denoted with prefix `ENSG:`

#### Organ (UBERON)

Each Tabula Sapiens scRNA-seq file originates from a different organ, named in the prefix of the file name (passed with the `--name` argument in the preprocessing script). This (unexactly) corresponds to a tissue biosample that the AlphaGenome model has been conditioned on, which can be passed in its `ontology_terms` argument. A map from biosample name to UBERON ID is kept for the purpose of automatically initializing the tissue condition when running AlphaGenome on a given organ's Tabula Sapiens dataset.

#### Cell Type (Cell Ontology)

Each Tabula Sapiens scRNA-seq file has annotated cell types using Cell Ontology. This consists of a cell ontology ID and a cell ontology class, which is the readable corresponding identifier. A map between these is stored for each Tabula Sapiens dataset (organ) processed.

AlphaGenome was also conditioned on certain cell types. It is expected that not all cell types in scRNA-seq data appear in the bulk RNA-seq data used to train AlphaGenome (this is demonstrated in `alphagenome_preprocessing.ipynb`, showing a large motivation behind this project), likely only the more common ones. Due to this difference, a list of supported cell types by AlphaGenome is also kept, derived from the model's metadata.

#### Gene ID (ENSEMBL)

Each gene in the Tabula Sapiens scRNA-seq dataset has a name and an ID, which is ENSG (ENS for ENSEMBL, G for gene) followed by a numeric sequence.

AlphaGenome takes as input a genomic interval defined by chromosome, start index, end index, and strand. To resolve each ENSEMBL ID to these coordinates, the preprocessed expression data is joined with AlphaGenome's own reference annotation: a GTF file for the hg38 human reference genome (hosted by them). Note that Tabula Sapiens is also hg38-based. The GTF file is filtered to gene features and is indexed by the same ENSEMBL IDs. Note that genes in the expression dataset without a matching entry in the GTF are dropped.