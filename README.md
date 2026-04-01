# scAlphaGenome

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Load environment variables**
```bash
source env
```

**3. Authenticate with GCP**
```bash
gcloud auth application-default login
gcloud config set project $GCP_PROJECT
```

## Data Preprocessing

**1. Create local data directories** (if not already done)
```bash
mkdir -p data/raw data/processed
```

**2. Download data** from [Tabula Sapiens v2](https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984) and place `.h5ad` files in `data/raw/`.

Currently processed organs:
* Heart
* Lung

**3. Run preprocessing script**

```bash
uv run python3 preprocess.py \
    --input  <filename>.h5ad \
    --output <output_name>.parquet \
    --assay  <10X|smartseq> \
    --gcs
```

Example:
```bash
uv run python3 preprocess.py \
    --input  Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad \
    --output lung_10x_pseudobulk.parquet \
    --assay  10X \
    --gcs
```

Output is saved locally to `data/processed/` and uploaded to `gs://scalphagenome-data/preprocessed/`. Omit `--gcs` to skip the upload.
