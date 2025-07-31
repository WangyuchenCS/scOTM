# 🚀 scOTM: A Deep Learning Framework for Predicting Single-Cell Perturbation Responses with Large Language Models

**Modeling drug-induced transcriptional responses at the single-cell level is essential for advancing human healthcare, particularly in understanding disease mechanisms, assessing therapeutic efficacy, and anticipating adverse effects. We propose scOTM, a deep learning framework designed to predict single-cell perturbation responses from unpaired data, focusing on generalization to unseen cell types. scOTM integrates prior biological knowledge of perturbations and cellular states, derived from large language models specialized for molecular and single-cell corpora.**

> This repository contains code and pretrained resources for our paper:
> **"\[scOTM: A Deep Learning Framework for Predicting Single-Cell Perturbation Responses with Large Language Models]"**
> \[Yuchen Wang et al.]
<!-- > \[Journal / Conference, Year]
> \[DOI or arXiv Link] -->

---

## 📖 Overview

This project provides reproducible setup for results in the paper, including:

* Environment Setup
* Preprocessing and handling of single-cell RNA-seq data
* Extraction of LLM embeddings
* Model training and evaluation scripts

---

## 💼 Repository Structure

```
├── data/                   # Input datasets (.h5ad)
├── LLM/                    # code for extracting embedding of LLM models
├── src/                    # Training, evaluation, utility scripts
├── env/                    # Environment file
├── tutorial.ipynb          # Full tutorial
└── README.md

```

---

## ⚙️ Environment Setup

This project is built with Conda and Python 3.8+. We recommend using the provided `.yml` file to fully replicate the environment.

### ✅ Install via Conda

```bash
conda env create -f scOTM.yml
conda activate scOTM
```

---

## 🌟 Extracting Embeddings from scGPT

This project requires **cell-level embeddings** extracted from the pretrained [scGPT](https://github.com/bowang-lab/scGPT) model.

### ✅ Step 1: Install scGPT

```bash
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT
pip install -e .
```

### ⏬ Step 2: Download Pretrained Models

```bash
# Example: replace with correct IDs or links
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1x1SfmFdI-zcocmqWAd7ZTC9CTEAVfKZq' -O best_model.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1jfT_T5n8WNbO9QZcLWObLdRG8lYFKH-Q' -O vocab.json
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15TEZmd2cZCrHwgfE424fgQkGUZCXiYrR' -O args.json
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

smaple_data_path = '../data/PBMC.h5ad'
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


## 🧬 Extracting Drug or Molecular Embeddings

We support extracting molecular embeddings from **SMILES strings** using **ChemBERTa**, and protein embeddings from **FASTA sequences** using **ESM2**.

###  1. ChemBERTa Embeddings (for Drug)

Install ChemBERTa dependencies:

```bash
pip install transformers rdkit pandas
```

Run embedding extraction:

```bash
python LLM/extract_drug_embedding.py \
    --smiles "CCCC" \  
    --model_name "seyonec/ChemBERTa-zinc-base-v1" \
    --output_path drug_embedding.npy
```
note that change smiles according to your needs.

###  2. ESM2 Embeddings (for Molecular)

Install ESM2 dependencies:

```bash
pip install fair-esm biopython torch
```

Run embedding extraction:

```bash
python LLM/extract_molecular_embedding.py \
    --uniprot_id P01574 \
    --model_name esm2_t33_650M_UR50D \
    --output_path molecular_embedding.npy

```

The output will be a `(1, embedding_dim)` tensor for single molecular.



---

## 🔥 Model Training & Evaluation

After preparing the embeddings, run model training:

```bash
python src/main.py
```
script including:

🧬 Loading and preprocessing scRNA-seq perturbation datasets
🧠 Initializing and training the scREPA model
🔮 Predicting perturbation responses for unseen cell types or conditions
📊 Visualizing and evaluating model performance

A complete training and evaluation pipeline is provided in the tutorial.ipynb

---


## 📄 Citation

If you find this work useful, please cite:

```bibtex
comming soon
```

---

## 👏 Acknowledgements

This project builds upon the excellent works, including:

- [scGPT](https://github.com/bowang-lab/scGPT) for cell embedding
- [ESM2](https://github.com/facebookresearch/esm) for protein sequence embeddings
- [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) for molecular embeddings from SMILES

---

## 📬 Contact

For questions or contributions, feel free to open an issue.