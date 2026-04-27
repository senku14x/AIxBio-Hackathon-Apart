"""
AIxBio Hackathon 2026 - Model Scaling Evaluation (ESM-2 3B)
===========================================================
Evaluates whether scaling up the model capacity to 3 Billion parameters 
resolves the phylogenetic confounding observed in the 650M model. 

Applies the exact same zero-shot cross-family holdout split to the 3B 
representations, sweeps the checkpoint layers to find the optimal validation 
representation, and evaluates final generalization on the withheld families.

Usage:
  python evaluate_3b.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

random.seed(42)
np.random.seed(42)

DATA_DIR = Path("./data")

# 3-Family Zero-Shot Holdout
HOLDOUT_EXACT = {
    "poxvirus": ["Monkeypox virus", "Ectromelia virus"],
    "enterobacteriaceae":["Yersinia pestis", "UniProt:Yersinia pestis virulence", "Yersinia pseudotuberculosis"],
    "coronavirus": ["MERS coronavirus", "Bat coronavirus RaTG13"],
}

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print(" ESM-2 3B MODEL EVALUATION (Cross-Family Generalization)")
    print("="*60)

    # 1. Load 3B Representations
    print("Loading 3B representations...")
    with h5py.File(DATA_DIR / "representations_3B.h5", "r") as f:
        reps = f["representations"]
        seq_ids =[s.decode() if isinstance(s, bytes) else s for s in f["seq_ids"][:]]
        X = np.stack([reps[sid][:] for sid in seq_ids], axis=0)  # Shape: (N, 7, 2560)
        labels = f["labels"][:]
        layer_indices = f["layer_indices"][:]

    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}

    # 2. Apply Zero-Shot Split
    holdout_set = set(org for orgs in HOLDOUT_EXACT.values() for org in orgs)

    test_idx, other_idx = [],[]
    for i, sid in enumerate(seq_ids):
        org = id_to_meta.get(sid, {}).get("organism", "")
        if org in holdout_set:
            test_idx.append(i)
        else:
            other_idx.append(i)

    other_labels = labels[other_idx]
    val_idx, train_idx = [], []
    for lbl in [0, 1]:
        lbl_indices =[other_idx[j] for j, l in enumerate(other_labels) if l == lbl]
        random.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * 0.10))
        val_idx.extend(lbl_indices[:n_val])
        train_idx.extend(lbl_indices[n_val:])

    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)

    # 3. Layer Sweep on Validation Set
    print("Sweeping checkpoint layers for optimal representation...")
    best_val_auroc = 0
    best_layer_idx = 0

    for i, phys_layer in enumerate(layer_indices):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx, i, :])
        X_v  = scaler.transform(X[val_idx, i, :])
        
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, labels[train_idx])
        val_auroc = roc_auc_score(labels[val_idx], clf.predict_proba(X_v)[:, 1])
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_layer_idx = i

    print(f"  Best Layer on Validation: Physical Layer {layer_indices[best_layer_idx]} (Val AUROC: {best_val_auroc:.4f})")

    # 4. Final Evaluation on Withheld Families
    print("\nEvaluating on withheld taxonomic families...")
    scaler = StandardScaler()
    X_tr_final = scaler.fit_transform(X[train_idx, best_layer_idx, :])
    X_te_final = scaler.transform(X[test_idx, best_layer_idx, :])

    clf_final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_final.fit(X_tr_final, labels[train_idx])

    test_preds = clf_final.predict(X_te_final)
    test_probs = clf_final.predict_proba(X_te_final)[:, 1]

    test_auroc = roc_auc_score(labels[test_idx], test_probs)
    fnr = (labels[test_idx] & (test_preds == 0)).sum() / labels[test_idx].sum()

    print(f"\nFINAL 3B TEST METRICS:")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Test FNR:   {fnr:.4f}")

    if test_auroc < 0.75:
        print("\n[Conclusion] Scaling to 3B parameters does not resolve phylogenetic confounding.")
    else:
        print("\n[Conclusion] Scaling to 3B parameters successfully generalizes across families.")

if __name__ == "__main__":
    main()
