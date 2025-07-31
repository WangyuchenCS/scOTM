# 🚀 scOTM: A Deep Learning Framework for Predicting Single-Cell Perturbation Responses with Large Language Models

**Modeling drug-induced transcriptional responses at the single-cell level is essential for advancing human healthcare, particularly in understanding disease mechanisms, assessing therapeutic efficacy, and anticipating adverse effects. We propose scOTM, a deep learning framework designed to predict single-cell perturbation responses from unpaired data, focusing on generalization to unseen cell types. scOTM integrates prior biological knowledge of perturbations and cellular states, derived from large language models specialized for molecular and single-cell corpora.**

> This repository contains code and pretrained resources for our paper:
> **"\[scOTM: A Deep Learning Framework for Predicting Single-Cell Perturbation Responses with Large Language Models]"**
> \[Yuchen Wang et al.]
<!-- > \[Journal / Conference, Year]
> \[DOI or arXiv Link] -->

---

## 📖 Overview

This project provides:

* Environment Setup
* Preprocessing and handling of single-cell RNA-seq data
* Extraction of LLM embeddings
* Model training and evaluation scripts
* Reproducible setup for results in the paper

---

## 📁 Repository Structure

```
├── data/                        # Input datasets (.h5ad)
├── checkpoints/                # Pretrained models (e.g., scGPT)
├── scripts/                    # Training, evaluation, utility scripts
├── results/                    # Output embeddings, metrics, figures
├── extract_scgpt_embedding.py  # Optional: embedding script
├── scPerturbation.yml          # Full Conda + pip environment file
├── requirements.txt            # pip fallback
└── README.md
```

---

## ⚙️ Environment Setup

This project is built with Conda and Python 3.8+. We recommend using the provided `.yml` file to fully replicate the environment.

### ✅ Install via Conda

```bash
conda env create -f scPerturbation.yml
conda activate scPerturbation
```

---

## 🧬 Extracting Cell Embeddings using scGPT

This project requires **cell-level embeddings** extracted using the pretrained [scGPT](https://github.com/bowang-lab/scGPT) model.

### ✅ Step 1: Install scGPT

```bash
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT
pip install -e .
```

### 📅 Step 2: Download Pretrained Models

```bash
# Example: replace with correct IDs or links
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1x1SfmFdI-zcocmqWAd7ZTC9CTEAVfKZq' -O best_model.pth
wget  --no-check-certificate 'https://drive.google.com/uc?export=download&id=1jfT_T5n8WNbO9QZcLWObLdRG8lYFKH-Q' -O vocab.json
wget  --no-check-certificate 'https://drive.google.com/uc?export=download&id=15TEZmd2cZCrHwgfE424fgQkGUZCXiYrR' -O args.json
```

### 📊 Step 3: Preprocess Input Expression Data

The input must be an .h5ad file (AnnData) with the following:
Gene expression matrix (adata.X) should be log-normalized counts or CPM
Genes must be aligned to the vocabulary used by the pretrained model
Gene names must match vocab.json keys

```python
import scanpy as sc
import numpy as np
import sys

smaple_data_path = '../../data/adata.h5ad'
adata = sc.read_h5ad(smaple_data_path)

gene_col = "Gene Symbol"
cell_type_key = "celltype"
batch_key = "tech"
N_HVG = 1800
adata.var[gene_col] = adata.var.index.values
# Make a copy of the dataset
org_adata = adata.copy()
# Preprocess the dataset and select N_HVG highly variable genes for downstream analysis.
# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

```

### 🧠 Step 4: Extract Embeddings

```python
# Generate the cell embeddings
# Now we will generate the cell embeddings for the dataset using embed_data function. embed_data calculates the cell embedding for each cell with the given scGPT model. The extracted embedding is stored in the X_scGPT field of obsm in AnnData.
import scgpt as scg
embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)
# Result: embed_adata.obsm["X_scGPT"]
```



## 🎉 Extracting Molecular and Protein Embeddings

We support extracting molecular embeddings from **SMILES strings** using **ChemBERTa**, and protein embeddings from **FASTA sequences** using **ESM2**.

###  1. ChemBERTa Embeddings (for Molecules)

Install ChemBERTa dependencies:

```bash
pip install transformers rdkit pandas
```

Run embedding extraction:

```bash
python scripts/extract_molecule_embedding.py \
    --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --model_name "seyonec/ChemBERTa-zinc-base-v1" \
    --output_path results/mol_embedding.npy
```

You can also provide a `.csv` file with a `smiles` column for batch extraction.

### 🔬 2. ESM2 Embeddings (for Proteins)

Install ESM2 dependencies:

```bash
pip install fair-esm biopython torch
```

Run embedding extraction:

```bash
python scripts/extract_protein_embedding.py \
    --fasta data/example_protein.fasta \
    --model_name esm2_t33_650M_UR50D \
    --output_path results/protein_embedding.pt
```

The output will be a `(1, embedding_dim)` tensor for single sequences, or batched for multiple.

---










---

## 🚀 Model Training & Evaluation

After preparing the embeddings, run model training:

```bash
python scripts/train_model.py \
    --data_path data/processed.h5ad \
    --embedding_key X_scGPT \
    --output_dir results/ \
    --batch_size 128 \
    --epochs 100
```

> Replace `train_model.py` and args as needed for your method.

---

## 📈 Results & Reproducibility

To reproduce the results in the paper:

1. Download raw data or use provided `.h5ad` files
2. Run preprocessing and scGPT embedding
3. Train the model using provided scripts

Results will be saved to `results/` including metrics and plots.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{YourCitation2025,
  title={Your Paper Title},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025},
  doi={...}
}
```

---

## 😋 Acknowledgements

This project builds upon the excellent [scGPT](https://github.com/bowang-lab/scGPT) model from Bo Wang's lab.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact:

* **Your Name** – [your.email@domain.edu](mailto:your.email@domain.edu)
* \[Your institution/lab page]
