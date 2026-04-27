"""
AIxBio Hackathon 2026 - Local Probe Evasion Test
================================================
Evaluates the robustness of family-specific (local) ESM-2 probes against 
adversarial sequence mutations designed to evade homology-based screening. 

This script compares the True Positive Rate (TPR) of local linear probes 
against standard BLASTp detection on pathogenic sequences subjected to 
random amino acid substitutions (0% to 30%).

Usage:
  python evasion_test.py
"""

import os
import random
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import torch
from transformers import EsmModel, EsmTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("./data")
BLAST_DB = "data/swissprot_db"
BEST_LAYER_POS = 21

TARGET_FAMILIES = ["clostridiaceae", "coronavirus"]
MUTATION_RATES =[0.0, 0.05, 0.10, 0.20, 0.30]

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def mutate_sequence(seq: str, mutation_rate: float) -> str:
    """Randomly substitute a specified fraction of residues in a sequence."""
    seq_list = list(seq)
    n_mut = max(1, int(len(seq_list) * mutation_rate))
    positions = random.sample(range(len(seq_list)), n_mut)
    for pos in positions:
        original = seq_list[pos]
        choices = [aa for aa in AA_ALPHABET if aa != original]
        seq_list[pos] = random.choice(choices)
    return "".join(seq_list)

def run_blast_on_seqs(sequences: list[str], db_path: str = BLAST_DB, evalue: float = 1e-5) -> list[bool]:
    """Run BLASTp against a local database and return a boolean list of flagged hits."""
    SELECT_AGENT_KEYWORDS =["botulinum", "neurotoxin", "sars", "mers", "coronavirus", "covid"]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        tmp_fasta = f.name
        for j, seq in enumerate(sequences):
            f.write(f">mut_{j}\n{seq}\n")
            
    tmp_out = tmp_fasta.replace(".fasta", "_blast.tsv")

    cmd =[
        "blastp", "-query", tmp_fasta, "-db", db_path,
        "-out", tmp_out, "-evalue", str(evalue),
        "-outfmt", "6 qseqid stitle", "-num_threads", "8", "-max_target_seqs", "3"
    ]
    subprocess.run(cmd, check=True)

    flagged = set()
    with open(tmp_out) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: 
                continue
            qid, stitle = parts[0], parts[1]
            if any(kw in stitle.lower() for kw in SELECT_AGENT_KEYWORDS):
                flagged.add(qid)

    os.unlink(tmp_fasta)
    os.unlink(tmp_out)
    return [f"mut_{i}" in flagged for i in range(len(sequences))]

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print(" AIxBio Evasion Test: Local Probes vs BLAST")
    print("="*60)
    print("Initializing ESM-2 model and loading data...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", 
        output_hidden_states=True, 
        add_pooling_layer=False
    ).to(device).eval()

    with h5py.File(DATA_DIR / "representations_650M.h5", "r") as f:
        reps = f["representations"]
        seq_ids =[s.decode() if isinstance(s, bytes) else s for s in f["seq_ids"][:]]
        X = np.stack([reps[sid][BEST_LAYER_POS, :] for sid in seq_ids], axis=0)
        labels = f["labels"][:]
        
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}
    
    fasta_path = DATA_DIR / "sequences.fasta"
    records = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}

    for fam in TARGET_FAMILIES:
        print(f"\n[Target Family: {fam.upper()}]")
        
        # Train Local Probe
        fam_idx =[i for i, sid in enumerate(seq_ids) if id_to_meta[sid].get("family") == fam]
        X_fam, y_fam = X[fam_idx], labels[fam_idx]
        
        scaler = StandardScaler()
        X_fam_scaled = scaler.fit_transform(X_fam)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_fam_scaled, y_fam)

        # Retrieve positive sequences
        pos_idx =[i for i in fam_idx if labels[i] == 1]
        pos_seqs = [records[seq_ids[i]] for i in pos_idx if seq_ids[i] in records]
        
        if not pos_seqs:
            print(f"  No positive sequences found for {fam}. Skipping.")
            continue

        print(f"  Evaluating {len(pos_seqs)} pathogenic sequences across mutation thresholds...")
        
        for rate in MUTATION_RATES:
            mut_seqs =[mutate_sequence(s, rate) for s in pos_seqs]
            
            all_embs =[]
            for i in range(0, len(mut_seqs), 32):
                batch = [s[:1022] for s in mut_seqs[i:i+32]]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    
                hs = outputs.hidden_states[BEST_LAYER_POS]
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (hs * mask).sum(1) / mask.sum(1)
                all_embs.append(emb.cpu().float().numpy())
            
            X_mut = np.vstack(all_embs)
            
            probe_preds = clf.predict(scaler.transform(X_mut))
            probe_tpr = np.mean(probe_preds)
            
            blast_flags = run_blast_on_seqs(mut_seqs)
            blast_tpr = np.mean(blast_flags)
            
            print(f"    Mutation {int(rate*100):02d}% | Local Probe TPR: {probe_tpr:.3f} | BLAST TPR: {blast_tpr:.3f}")

    print("\nEvasion evaluation complete.")

if __name__ == "__main__":
    main()
