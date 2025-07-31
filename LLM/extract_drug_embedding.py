import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def extract_drug_embedding(smiles: str, model_name: str, output_path: str):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Tokenize SMILES input
    inputs = tokenizer(smiles, return_tensors="pt")

    # Forward pass without gradient
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token embedding (first token)
    embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_dim)

    # Save as numpy file
    np.save(output_path, embedding.cpu().numpy())
    print(f"Embedding saved to {output_path} with shape {embedding.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ChemBERTa embedding for a SMILES string")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES string")
    parser.add_argument("--model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1", help="Hugging Face model name")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output embedding (.npy)")
    args = parser.parse_args()

    extract_drug_embedding(args.smiles, args.model_name, args.output_path)