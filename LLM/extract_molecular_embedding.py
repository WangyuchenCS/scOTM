import torch
import esm
import numpy as np
from Bio import SeqIO
from io import StringIO
import requests
import argparse
import os


def fetch_uniprot_fasta(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url)
    if r.status_code == 200:
        return r.text
    else:
        raise ValueError(f"Failed to fetch UniProt sequence for {uniprot_id}")


def extract_esm2_embedding(sequence, name, model, batch_converter):
    batch_labels, batch_strs, batch_tokens = batch_converter([(name, sequence)])
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1].mean(0).cpu().numpy()  # mean over residues
    return embedding


def main(uniprot_id, model_name, output_path):
    # Load sequence from UniProt
    fasta = fetch_uniprot_fasta(uniprot_id)
    seq_record = list(SeqIO.parse(StringIO(fasta), "fasta"))[0]
    sequence = str(seq_record.seq)

    # Load ESM2 model
    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # Extract embedding
    embedding = extract_esm2_embedding(sequence, seq_record.id, model, batch_converter)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    print(f"Saved embedding to {output_path} with shape {embedding.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ESM2 embedding for a UniProt protein")
    parser.add_argument("--uniprot_id", type=str, required=True, help="UniProt accession ID")
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D", help="Model name")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the .npy embedding")
    args = parser.parse_args()

    main(args.uniprot_id, args.model_path, args.model_name, args.output_path)
