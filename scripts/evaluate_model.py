"""
AIxBio Hackathon 2026 - Zero-Shot Evaluation Pipeline
=====================================================
Executes the core zero-shot cross-family evaluation against the BLAST baseline.

Steps:
  1. Loads pre-extracted ESM-2 embeddings (650M or 3B).
  2. Applies the strict 3-family zero-shot holdout split.
  3. Sweeps all layers to find the optimal pathogenic representation.
  4. Trains a linear probe on the training families.
  5. Evaluates on the held-out test families and compares False Negative 
     Rates against standard Swiss-Prot BLASTp matching.

Usage:
  python evaluate_model.py --model_size 650M
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from extract_representations import load_hdf5
from probing import (
    probe_all_layers, 
    train_classifier, 
    evaluate_classifier_on_test, 
    run_blast_on_test,
    plot_classifier_vs_blast
)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./results")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# The strict zero-shot cross-family holdout used in the paper
HOLDOUT_EXACT = {
    "poxvirus":["Monkeypox virus", "Ectromelia virus"],
    "enterobacteriaceae":["Yersinia pestis", "UniProt:Yersinia pestis virulence", "Yersinia pseudotuberculosis"],
    "coronavirus":["MERS coronavirus", "Bat coronavirus RaTG13"],
}

# -----------------------------------------------------------------------------
# MAIN EVALUATION
# -----------------------------------------------------------------------------

def main(model_size: str):
    print("\n" + "=" * 60)
    print(f" ZERO-SHOT EVALUATION: ESM-2 {model_size}")
    print("=" * 60)

    # 1. Load Data
    h5_filename = f"representations_{model_size}.h5"
    reps, labels, seq_ids, layer_indices = load_hdf5(DATA_DIR / h5_filename)
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")

    # 2. Apply Zero-Shot Split
    holdout_set = set(org for orgs in HOLDOUT_EXACT.values() for org in orgs)
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}

    test_idx, other_idx = [],[]
    for i, sid in enumerate(seq_ids):
        organism = id_to_meta.get(sid, {}).get("organism", "")
        if organism in holdout_set:
            test_idx.append(i)
        else:
            other_idx.append(i)

    # Stratified 90/10 Val Split
    other_labels = labels[other_idx]
    val_idx, train_idx = [], []
    for lbl in [0, 1]:
        lbl_indices = [other_idx[j] for j, l in enumerate(other_labels) if l == lbl]
        random.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * 0.10))
        val_idx.extend(lbl_indices[:n_val])
        train_idx.extend(lbl_indices[n_val:])

    split = {
        "train_idx": np.array(train_idx),
        "val_idx":   np.array(val_idx),
        "test_idx":  np.array(test_idx),
        "id_to_meta": id_to_meta,
    }

    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_idx)} sequences (Pos: {labels[train_idx].sum()}, Neg: {(labels[train_idx]==0).sum()})")
    print(f"  Val:   {len(val_idx)} sequences (Pos: {labels[val_idx].sum()}, Neg: {(labels[val_idx]==0).sum()})")
    print(f"  Test:  {len(test_idx)} sequences (Pos: {labels[test_idx].sum()}, Neg: {(labels[test_idx]==0).sum()})")

    # 3. Layer Sweep & Training
    probe_df = probe_all_layers(reps, labels, split, layer_indices)
    best_layer_pos = int(probe_df.loc[probe_df["auroc"].idxmax(), "layer_pos"])

    lr_clf, mlp_clf, scaler = train_classifier(reps, labels, split, best_layer_pos)

    # 4. Final Zero-Shot Evaluation
    clf_results = evaluate_classifier_on_test(
        reps, labels, split, best_layer_pos, lr_clf, mlp_clf, scaler
    )

    # 5. BLAST Baseline Comparison
    fasta_path = DATA_DIR / "sequences.fasta"
    test_ids =[seq_ids[i] for i in split["test_idx"]]
    y_test = labels[split["test_idx"]]
    
    blast_flags = run_blast_on_test(seq_ids, split["test_idx"], fasta_path)
    
    tp = sum(1 for i, sid in enumerate(test_ids) if sid in blast_flags and y_test[i] == 1)
    fn = sum(1 for i, sid in enumerate(test_ids) if sid not in blast_flags and y_test[i] == 1)
    blast_tpr = tp / (tp + fn) if (tp + fn) else 0
    blast_fnr = fn / (tp + fn) if (tp + fn) else 0
    
    print(f"\n  BLAST False Negative Rate: {blast_fnr:.4f}")

    # Plot
    plot_classifier_vs_blast(clf_results, blast_tpr, blast_fnr)
    print(f"\nPipeline complete. Artifacts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ESM-2 Models")
    parser.add_argument("--model_size", default="650M", choices=["650M", "3B"])
    args = parser.parse_args()
    main(args.model_size)
