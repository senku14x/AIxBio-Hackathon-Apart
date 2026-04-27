"""
AIxBio Hackathon 2026 - Interpretability Robustness Suite
=========================================================
Executes a suite of four falsification and robustness tests to evaluate 
the structural vs. phylogenetic properties of ESM-2 representations:

  1. UMAP Visualization: Maps representational geometry (Taxonomy vs Risk).
  2. Intra-Family Probing: Evaluates risk separation within local phylogenies.
  3. Fragment Truncation: Tests representation stability on short synthesis orders.
  4. OOD Artifact Detection: Tests False Positive Rate on scrambled/nonsense sequences.

Usage:
  python robustness_suite.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import torch
from transformers import EsmModel, EsmTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from Bio import SeqIO

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_LAYER_POS = 21 

# Replicate the zero-shot cross-family holdout
HOLDOUT_EXACT = {
    "poxvirus": ["Monkeypox virus", "Ectromelia virus"],
    "enterobacteriaceae":["Yersinia pestis", "UniProt:Yersinia pestis virulence", "Yersinia pseudotuberculosis"],
    "coronavirus": ["MERS coronavirus", "Bat coronavirus RaTG13"],
}

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_and_split():
    """Load embeddings and reproduce the exact zero-shot data split."""
    with h5py.File(DATA_DIR / "representations_650M.h5", "r") as f:
        reps = f["representations"]
        seq_ids = [s.decode() if isinstance(s, bytes) else s for s in f["seq_ids"][:]]
        # Load only the targeted optimal layer to conserve memory
        X = np.stack([reps[sid][BEST_LAYER_POS, :] for sid in seq_ids], axis=0)
        labels = f["labels"][:]
        
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}
    
    holdout_set = set(org for orgs in HOLDOUT_EXACT.values() for org in orgs)
    
    test_idx, train_idx = [],[]
    for i, sid in enumerate(seq_ids):
        org = id_to_meta.get(sid, {}).get("organism", "")
        if org in holdout_set:
            test_idx.append(i)
        else:
            train_idx.append(i)
            
    return X, labels, np.array(train_idx), np.array(test_idx), seq_ids, id_to_meta

# -----------------------------------------------------------------------------
# 1. UMAP VISUALIZATION
# -----------------------------------------------------------------------------

def run_umap_visualization(X_test, y_test, test_idx, id_to_meta, seq_ids):
    """Generates a 2D UMAP projection of the representation space."""
    print("\n" + "="*60)
    print(" 1. UMAP VISUALIZATION")
    print("="*60)
    print("Fitting UMAP on test set representations...")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    embedding = reducer.fit_transform(X_test_scaled)
    
    families = [id_to_meta[seq_ids[i]]["family"] for i in test_idx]
    labels_text =["Pathogenic" if y == 1 else "Benign" for y in y_test]
    
    df_plot = pd.DataFrame({
        "UMAP 1": embedding[:, 0],
        "UMAP 2": embedding[:, 1],
        "Risk": labels_text,
        "Family": families
    })
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot, x="UMAP 1", y="UMAP 2", 
        hue="Family", style="Risk", s=100, alpha=0.8,
        palette="tab10", markers={"Benign": "o", "Pathogenic": "X"}
    )
    plt.title(f"ESM-2 Layer {BEST_LAYER_POS} Test Set Geometry\n(Clustering driven by taxonomy, not biosecurity risk)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "umap_taxonomy_vs_risk.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# 2. INTRA-FAMILY (LOCAL) PROBING
# -----------------------------------------------------------------------------

def run_intra_family_probing(X, labels, id_to_meta, seq_ids):
    """Evaluates risk separability within individual taxonomic groups."""
    print("\n" + "="*60)
    print(" 2. INTRA-FAMILY (LOCAL EXPERT) PROBING")
    print("="*60)
    
    families = [id_to_meta[sid].get("family", "") for sid in seq_ids]
    df = pd.DataFrame({"idx": range(len(labels)), "family": families, "label": labels})
    counts = df.groupby(["family", "label"]).size().unstack(fill_value=0)
    
    # Filter for families with sufficient samples for 5-Fold CV
    valid_families = counts[(counts[0] >= 10) & (counts[1] >= 10)].index.tolist()
    print(f"Evaluating local probes via 5-Fold CV for families: {valid_families}\n")
    
    for fam in valid_families:
        fam_idx = df[df["family"] == fam]["idx"].values
        X_fam = X[fam_idx]
        y_fam = labels[fam_idx]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs =[]
        
        for train_i, test_i in cv.split(X_fam, y_fam):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_fam[train_i])
            X_te = scaler.transform(X_fam[test_i])
            
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_tr, y_fam[train_i])
            preds = clf.predict_proba(X_te)[:, 1]
            aurocs.append(roc_auc_score(y_fam[test_i], preds))
            
        print(f"  {fam:<20} Local AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")

# -----------------------------------------------------------------------------
# 3 & 4. FRAGMENT & OOD ARTIFACT TESTS
# -----------------------------------------------------------------------------

def run_esm_extraction(sequences, device, model, tokenizer):
    """Helper to dynamically re-extract embeddings for modified sequences."""
    batch_size = 32
    all_reps =[]
    
    for i in range(0, len(sequences), batch_size):
        batch = [s[:1022] for s in sequences[i:i+batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        hs = outputs.hidden_states[BEST_LAYER_POS]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (hs * mask).sum(1) / mask.sum(1)
        all_reps.append(emb.cpu().float().numpy())
        
    return np.vstack(all_reps)

def test_fragments_and_scrambles(X_train, y_train, test_idx, seq_ids, labels, labels_df):
    """Evaluates probe vulnerability to short sequences and OOD noise."""
    print("\n" + "="*60)
    print(" 3 & 4. FRAGMENT & OOD ARTIFACT TESTS")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    print("Loading ESM-2 650M on device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", output_hidden_states=True, add_pooling_layer=False).to(device).eval()

    fasta_path = DATA_DIR / "sequences.fasta"
    records = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    
    # ---------------------------------------------------------
    # TEST 3: Short Fragment Truncation (CBAI Track Vulnerability)
    # ---------------------------------------------------------
    print("\n[Test 3] Fragment Robustness (Truncating Pathogens to 60 AA)")
    pos_test_idx = [i for i in test_idx if labels[i] == 1]
    pos_seqs = [records[seq_ids[i]] for i in pos_test_idx if seq_ids[i] in records]
    
    full_embs = run_esm_extraction(pos_seqs, device, model, tokenizer)
    full_preds = clf.predict(scaler.transform(full_embs))
    full_recall = np.mean(full_preds)
    
    frag_seqs = [s[:60] for s in pos_seqs]
    frag_embs = run_esm_extraction(frag_seqs, device, model, tokenizer)
    frag_preds = clf.predict(scaler.transform(frag_embs))
    frag_recall = np.mean(frag_preds)
    
    print(f"  Full Sequence TPR (Recall): {full_recall:.4f}")
    print(f"  Fragment Sequence TPR:      {frag_recall:.4f}")
    if frag_recall < full_recall - 0.2:
        print("  [Vulnerability Detected] Significant recall degradation on short sequences.")
    
    # ---------------------------------------------------------
    # TEST 4: Scrambled Benign Test (OOD Artifact Check)
    # ---------------------------------------------------------
    print("\n[Test 4] Scrambled Benign Sequences (OOD Artifact Check)")
    neg_test_idx =[i for i in test_idx if labels[i] == 0]
    
    if len(neg_test_idx) > 100: 
        neg_test_idx = random.sample(neg_test_idx, 100)
    neg_seqs =[records[seq_ids[i]] for i in neg_test_idx if seq_ids[i] in records]
    
    full_neg_embs = run_esm_extraction(neg_seqs, device, model, tokenizer)
    full_neg_preds = clf.predict(scaler.transform(full_neg_embs))
    full_fpr = np.mean(full_neg_preds)
    
    scrambled_seqs =["".join(random.sample(s, len(s))) for s in neg_seqs]
    scram_embs = run_esm_extraction(scrambled_seqs, device, model, tokenizer)
    scram_preds = clf.predict(scaler.transform(scram_embs))
    scram_fpr = np.mean(scram_preds)
    
    print(f"  Normal Benign FPR:    {full_fpr:.4f}")
    print(f"  Scrambled Benign FPR: {scram_fpr:.4f}")
    if scram_fpr > full_fpr + 0.2:
        print("  [Artifact Detected] High FPR on scrambled sequences suggests OOD sensitivity.")
    else:
        print("[Artifact Cleared] Model correctly ignores scrambled noise.")

    print("\nRobustness evaluation complete.")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    X, labels, train_idx, test_idx, seq_ids, id_to_meta = load_and_split()
    
    run_umap_visualization(X[test_idx], labels[test_idx], test_idx, id_to_meta, seq_ids)
    run_intra_famil
