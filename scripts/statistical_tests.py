"""
AIxBio Hackathon 2026 - Statistical Significance Testing
========================================================
Performs formal statistical evaluation of the global ESM-2 linear probe 
against the standard BLAST homology baseline.

Calculates:
  1. Bootstrapped 95% Confidence Intervals for the ESM-2 Test AUROC.
  2. McNemar's Test (exact) p-value to determine if the difference in 
     classification accuracy between ESM-2 and BLAST is statistically significant.

Usage:
  python statistical_tests.py
"""

import os
import random
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from Bio import SeqIO

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

random.seed(42)
np.random.seed(42)

DATA_DIR = Path("./data")
BLAST_DB = "data/swissprot_db"
BEST_LAYER_POS = 21

HOLDOUT_EXACT = {
    "poxvirus":["Monkeypox virus", "Ectromelia virus"],
    "enterobacteriaceae":["Yersinia pestis", "UniProt:Yersinia pestis virulence", "Yersinia pseudotuberculosis"],
    "coronavirus": ["MERS coronavirus", "Bat coronavirus RaTG13"],
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def load_data():
    """Load embeddings and apply the zero-shot cross-family split."""
    with h5py.File(DATA_DIR / "representations_650M.h5", "r") as f:
        reps = f["representations"]
        seq_ids = [s.decode() if isinstance(s, bytes) else s for s in f["seq_ids"][:]]
        X = np.stack([reps[sid][BEST_LAYER_POS, :] for sid in seq_ids], axis=0)
        labels = f["labels"][:]
        
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}
    
    holdout_set = set(org for orgs in HOLDOUT_EXACT.values() for org in orgs)
    
    test_idx, train_idx = [],[]
    for i, sid in enumerate(seq_ids):
        if id_to_meta.get(sid, {}).get("organism", "") in holdout_set:
            test_idx.append(i)
        else:
            train_idx.append(i)
            
    return X, labels, np.array(train_idx), np.array(test_idx), seq_ids


def run_blast(seq_ids, test_idx):
    """Run baseline BLASTp keyword screening on test sequences."""
    fasta_path = DATA_DIR / "sequences.fasta"
    records = {r.id: r for r in SeqIO.parse(fasta_path, "fasta")}
    test_ids = [seq_ids[i] for i in test_idx]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        tmp_fasta = f.name
        for sid in test_ids:
            if sid in records:
                SeqIO.write(records[sid], f, "fasta")
                
    tmp_out = tmp_fasta.replace(".fasta", "_blast.tsv")
    cmd =[
        "blastp", "-query", tmp_fasta, "-db", BLAST_DB, "-out", tmp_out, 
        "-evalue", "1e-5", "-outfmt", "6 qseqid stitle", "-num_threads", "8", "-max_target_seqs", "3"
    ]
    subprocess.run(cmd, check=True)
    
    SELECT_AGENT_KEYWORDS =["pox", "pestis", "sars", "mers", "coronavirus", "variola"]
    flagged = set()
    with open(tmp_out) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            if any(kw in parts[1].lower() for kw in SELECT_AGENT_KEYWORDS):
                flagged.add(parts[0])
                
    os.unlink(tmp_fasta)
    os.unlink(tmp_out)
    return np.array([1 if sid in flagged else 0 for sid in test_ids])


def bootstrap_auroc(y_true, y_probs, n_bootstraps=1000):
    """Calculates 95% Confidence Interval for AUROC via bootstrapping."""
    bootstrapped_scores =[]
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return ci_lower, ci_upper

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print(" STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)
    
    X, labels, train_idx, test_idx, seq_ids = load_data()
    
    # Train Global Probe
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    y_tr, y_te = labels[train_idx], labels[test_idx]
    
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    
    esm_probs = clf.predict_proba(X_te)[:, 1]
    esm_preds = clf.predict(X_te)
    blast_preds = run_blast(seq_ids, test_idx)
    
    # 1. Bootstrapped CIs
    ci_lower, ci_upper = bootstrap_auroc(y_te, esm_probs)
    esm_auroc = roc_auc_score(y_te, esm_probs)
    print(f"\n[1] ESM-2 Global Probe AUROC: {esm_auroc:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    
    # 2. McNemar's Test
    esm_correct = (esm_preds == y_te)
    blast_correct = (blast_preds == y_te)
    
    both_correct = np.sum(esm_correct & blast_correct)
    esm_only = np.sum(esm_correct & ~blast_correct)
    blast_only = np.sum(~esm_correct & blast_correct)
    neither = np.sum(~esm_correct & ~blast_correct)
    
    table = [[both_correct, esm_only],
             [blast_only, neither]]
             
    result = mcnemar(table, exact=True)
    
    print("\n[2] McNemar's Test (ESM-2 vs BLAST Classification Errors)")
    print(f"  ESM-2 Correct / BLAST Wrong: {esm_only}")
    print(f"  BLAST Correct / ESM-2 Wrong: {blast_only}")
    print(f"  p-value: {result.pvalue:.4e}")
    
    if result.pvalue < 0.05:
        print("[Conclusion] The performance difference is statistically significant.")
    else:
        print("  [Conclusion] NO SIGNIFICANT DIFFERENCE. The AI probe performs comparably to the keyword homology baseline.")

if __name__ == "__main__":
    main()
