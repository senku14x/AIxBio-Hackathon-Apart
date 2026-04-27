"""
AIxBio Hackathon 2026 - Global Probing & Falsification Pipeline
===============================================================
Evaluates the capacity of Protein Language Models (ESM-2, ProtTrans) to 
zero-shot detect biosecurity risk. Compares global linear probes against 
standard BLAST homology matching.

Pipeline stages:
  1. Zero-Shot Split: Holds out specific taxonomic families from training.
  2. Layer Sweep: Fits logistic regression probes across all ESM-2 layers.
  3. Classifier vs BLAST: Evaluates best layer against keyword-based BLASTp.
  4. Adversarial Robustness: Tests detection degradation under random mutations.
  5. Cross-Model Consistency: Compares ESM-2 geometry with ProtTrans-T5.

Usage:
  python probing.py
"""

import os
import json
import random
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, classification_report
)

import torch
from transformers import EsmModel, EsmTokenizer, T5Tokenizer, T5EncoderModel
from Bio import SeqIO

from extract_representations import load_hdf5

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Zero-Shot Cross-Family Holdout
# Three families are entirely withheld from the training/validation sets.
HOLDOUT_ORGANISMS = {
    "poxvirus":           ["Monkeypox virus", "Ectromelia virus"],
    "enterobacteriaceae":["Yersinia pestis", "UniProt:Yersinia pestis virulence", "Yersinia pseudotuberculosis"],
    "coronavirus":["MERS coronavirus", "Bat coronavirus RaTG13"],
}

MUTATION_RATES =[0.05, 0.10, 0.20, 0.30, 0.40]
N_ADVERSARIAL_SEQS = 30

BLAST_DB     = "data/swissprot_db"
BLAST_EVALUE = 1e-5


# -----------------------------------------------------------------------------
# 1. SPLIT GENERATION
# -----------------------------------------------------------------------------

def make_holdout_split(
    reps: np.ndarray,
    labels: np.ndarray,
    seq_ids: list[str],
    labels_df: pd.DataFrame,
) -> dict:
    """Build train/val/test indices using a zero-shot cross-family holdout."""
    id_to_meta = {row["seq_id"]: row.to_dict() for _, row in labels_df.iterrows()}

    holdout_set = set(org for orgs in HOLDOUT_ORGANISMS.values() for org in orgs)

    test_idx, other_idx = [],[]
    for i, sid in enumerate(seq_ids):
        organism = id_to_meta.get(sid, {}).get("organism", "")
        if organism in holdout_set:
            test_idx.append(i)
        else:
            other_idx.append(i)

    other_labels = labels[other_idx]
    val_frac = 0.10
    val_idx, train_idx = [], []

    for lbl in[0, 1]:
        lbl_indices = [other_idx[j] for j, l in enumerate(other_labels) if l == lbl]
        random.shuffle(lbl_indices)
        n_val = max(1, int(len(lbl_indices) * val_frac))
        val_idx.extend(lbl_indices[:n_val])
        train_idx.extend(lbl_indices[n_val:])

    print("\n" + "=" * 60)
    print(" SPLIT SUMMARY (Zero-Shot Cross-Family Holdout)")
    print("=" * 60)
    print(f" Train: {len(train_idx)} (pos={labels[train_idx].sum()}, neg={(labels[train_idx]==0).sum()})")
    print(f" Val:   {len(val_idx)} (pos={labels[val_idx].sum()}, neg={(labels[val_idx]==0).sum()})")
    print(f" Test:  {len(test_idx)} (pos={labels[test_idx].sum()}, neg={(labels[test_idx]==0).sum()})")

    return {
        "train_idx": np.array(train_idx),
        "val_idx":   np.array(val_idx),
        "test_idx":  np.array(test_idx),
        "id_to_meta": id_to_meta,
    }


# -----------------------------------------------------------------------------
# 2. LAYER-WISE LINEAR PROBING
# -----------------------------------------------------------------------------

def probe_all_layers(reps: np.ndarray, labels: np.ndarray, split: dict, layer_indices: np.ndarray) -> pd.DataFrame:
    """Train logistic regression probes at every extracted layer."""
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    results =[]

    print("\n" + "=" * 60)
    print(" LINEAR PROBING - LAYER SWEEP")
    print("=" * 60)

    for layer_pos in tqdm(range(reps.shape[1]), desc="Probing layers"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(reps[train_idx, layer_pos, :])
        X_val   = scaler.transform(reps[val_idx, layer_pos, :])
        y_train, y_val = labels[train_idx], labels[val_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_val)[:, 1]
        preds = clf.predict(X_val)

        auroc = roc_auc_score(y_val, probs)
        f1    = f1_score(y_val, preds)
        
        prec, rec, _ = precision_recall_curve(y_val, probs)
        idx_95 = np.where(rec >= 0.95)[0]
        p_at_95r = prec[idx_95[-1]] if len(idx_95) > 0 else 0.0

        results.append({
            "layer_pos": layer_pos,
            "physical_layer": int(layer_indices[layer_pos]),
            "auroc": auroc,
            "f1": f1,
            "p_at_95r": p_at_95r,
        })

    df = pd.DataFrame(results)
    best_row = df.loc[df["auroc"].idxmax()]
    print(f"\n Best layer: Physical {int(best_row['physical_layer'])} | Val AUROC={best_row['auroc']:.4f}")

    df.to_csv(OUTPUT_DIR / "layer_probe_results.csv", index=False)
    plot_layer_profile(df)
    return df


def plot_layer_profile(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["physical_layer"], df["auroc"], label="AUROC", marker="o", ms=4)
    ax.plot(df["physical_layer"], df["f1"], label="F1", marker="s", ms=4)
    ax.set_xlabel("ESM-2 Layer")
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Validation Performance Across Layers")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_profile.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 3. CLASSIFIER VS BLAST
# -----------------------------------------------------------------------------

def train_classifier(reps: np.ndarray, labels: np.ndarray, split: dict, best_layer_pos: int) -> tuple:
    """Train optimal LR and MLP models on the best performing layer."""
    train_idx, val_idx = split["train_idx"], split["val_idx"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(reps[train_idx, best_layer_pos, :])
    X_val   = scaler.transform(reps[val_idx, best_layer_pos, :])

    print("\n" + "=" * 60)
    print(f" CLASSIFIER TRAINING (Layer {best_layer_pos})")
    print("=" * 60)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    lr.fit(X_train, labels[train_idx])
    print(f" LR  Val AUROC: {roc_auc_score(labels[val_idx], lr.predict_proba(X_val)[:, 1]):.4f}")

    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, early_stopping=True, random_state=RANDOM_SEED)
    mlp.fit(X_train, labels[train_idx])
    print(f" MLP Val AUROC: {roc_auc_score(labels[val_idx], mlp.predict_proba(X_val)[:, 1]):.4f}")

    return lr, mlp, scaler


def evaluate_classifier_on_test(reps, labels, split, best_layer_pos, lr_clf, mlp_clf, scaler) -> dict:
    """Evaluate trained models on the zero-shot withheld test set."""
    X_test = scaler.transform(reps[split["test_idx"], best_layer_pos, :])
    y_test = labels[split["test_idx"]]

    results = {}
    print("\n" + "=" * 60)
    print(" FINAL TEST SET EVALUATION (Unseen Taxonomic Families)")
    print("=" * 60)

    for name, clf in[("LR", lr_clf), ("MLP", mlp_clf)]:
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        fnr = (y_test & (preds == 0)).sum() / y_test.sum()

        print(f" {name} Model:")
        print(f"   AUROC: {roc_auc_score(y_test, probs):.4f}")
        print(f"   FNR:   {fnr:.4f} (False Negative Rate)")
        results[name] = {"auroc": roc_auc_score(y_test, probs), "fnr": fnr, "f1": f1_score(y_test, preds)}

    return results


def run_blast_on_test(seq_ids: list[str], test_idx: np.ndarray, fasta_path: Path) -> dict:
    """Run standard BLASTp keyword matching on the test sequences."""
    SELECT_AGENT_KEYWORDS =[
        "ebola", "marburg", "variola", "monkeypox", "anthrax", "anthracis",
        "plague", "pestis", "tularensis", "botulinum", "neurotoxin",
        "burkholderia mallei", "pseudomallei", "melioidosis",
        "coxiella", "q fever", "h5n1", "avian influenza",
        "sars", "mers", "nipah", "ricin",
    ]

    all_records = {r.id: r for r in SeqIO.parse(fasta_path, "fasta")}
    test_ids =[seq_ids[i] for i in test_idx]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        tmp_fasta = f.name
        for sid in test_ids:
            if sid in all_records:
                SeqIO.write(all_records[sid], f, "fasta")

    tmp_out = tmp_fasta.replace(".fasta", "_blast.tsv")

    print("\n" + "=" * 60)
    print(" BLAST HOMOLOGY BASELINE")
    print("=" * 60)
    
    cmd =[
        "blastp", "-query", tmp_fasta, "-db", BLAST_DB, "-out", tmp_out,
        "-evalue", str(BLAST_EVALUE), "-outfmt", "6 qseqid stitle",
        "-num_threads", "8", "-max_target_seqs", "3"
    ]
    subprocess.run(cmd, check=True)

    flagged = set()
    with open(tmp_out) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            if any(kw in parts[1].lower() for kw in SELECT_AGENT_KEYWORDS):
                flagged.add(parts[0])

    os.unlink(tmp_fasta)
    os.unlink(tmp_out)

    return flagged


def plot_classifier_vs_blast(clf_results: dict, blast_tpr: float, blast_fnr: float):
    methods =["LR", "MLP", "BLAST"]
    fnrs = [clf_results["LR"]["fnr"], clf_results["MLP"]["fnr"], blast_fnr]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(methods, fnrs, color=["#4C72B0", "#DD8452", "#55A868"], width=0.6)
    plt.title("False Negative Rate (Lower is Better)")
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, fnrs):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classifier_vs_blast.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 4. ADVERSARIAL MUTATION
# -----------------------------------------------------------------------------

def run_adversarial_experiment(labels, seq_ids, split, best_layer_pos, lr_clf, mlp_clf, scaler, fasta_path):
    """Evaluate probe robustness against random point mutations."""
    from transformers import EsmModel, EsmTokenizer
    import re

    print("\n" + "=" * 60)
    print(" ADVERSARIAL MUTATION EXPERIMENT")
    print("=" * 60)

    pos_test = [i for i in split["test_idx"] if labels[i] == 1]
    if len(pos_test) > N_ADVERSARIAL_SEQS:
        pos_test = random.sample(pos_test, N_ADVERSARIAL_SEQS)

    records = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    orig_seqs = [(seq_ids[i], records.get(seq_ids[i], "")) for i in pos_test]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", add_pooling_layer=False).to(device).eval()

    results =[]
    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

    for rate in MUTATION_RATES:
        mutated_seqs =[]
        for _, seq in orig_seqs:
            s_list = list(seq)
            n_mut = max(1, int(len(s_list) * rate))
            for pos in random.sample(range(len(s_list)), n_mut):
                s_list[pos] = random.choice([aa for aa in AA_ALPHABET if aa != s_list[pos]])
            mutated_seqs.append("".join(s_list))

        all_embs =[]
        for i in range(0, len(mutated_seqs), 16):
            batch = [s[:1022] for s in mutated_seqs[i:i+16]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                hs = model(**inputs, output_hidden_states=True).hidden_states[best_layer_pos]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            all_embs.append(((hs * mask).sum(1) / mask.sum(1)).cpu().numpy())

        X_mut = scaler.transform(np.vstack(all_embs))
        lr_rate = lr_clf.predict(X_mut).mean()
        
        print(f" Mutation {int(rate*100)}% | Probe TPR: {lr_rate:.3f}")
        results.append({"rate": rate, "lr_tpr": lr_rate})

    del model; torch.cuda.empty_cache()
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 5. CROSS-MODEL CONSISTENCY (ProtTrans)
# -----------------------------------------------------------------------------

def extract_prottrans(fasta_path: Path, labels_df: pd.DataFrame) -> tuple:
    """Extract representations using ProtTrans-T5 to verify architectural independence."""
    import re

    print("\n" + "=" * 60)
    print(" PROTTRANS-T5 CROSS-MODEL CONSISTENCY")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16).to(device).eval()

    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [r.id for r in records]
    sequences =[" ".join(list(re.sub(r"[UZOB]", "X", str(r.seq)[:1022].upper()))) for r in records]

    label_map = dict(zip(labels_df["seq_id"], labels_df["label"]))
    labels = np.array([label_map.get(sid, -1) for sid in seq_ids], dtype=np.int32)

    all_reps =[]
    for i in tqdm(range(0, len(sequences), 16), desc="ProtTrans Extraction"):
        inputs = tokenizer(sequences[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        all_reps.append(((outputs.last_hidden_state * mask).sum(1) / mask.sum(1)).cpu().float().numpy())

    del model; torch.cuda.empty_cache()
    return np.vstack(all_reps), labels, seq_ids


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print(" AIxBio Probing Pipeline")
    print("=" * 60)

    h5_path = DATA_DIR / "representations_650M.h5"
    fasta_path = DATA_DIR / "sequences.fasta"
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")

    reps, labels, seq_ids, layer_indices = load_hdf5(h5_path)

    # 1. Split
    split = make_holdout_split(reps, labels, seq_ids, labels_df)

    # 2. Probe Layers
    probe_df = probe_all_layers(reps, labels, split, layer_indices)
    best_pos = int(probe_df.loc[probe_df["auroc"].idxmax(), "layer_pos"])

    # 3. Classifiers
    lr_clf, mlp_clf, scaler = train_classifier(reps, labels, split, best_pos)
    clf_results = evaluate_classifier_on_test(reps, labels, split, best_pos, lr_clf, mlp_clf, scaler)

    # BLAST Comparison
    test_ids = [seq_ids[i] for i in split["test_idx"]]
    y_test = labels[split["test_idx"]]
    blast_flags = run_blast_on_test(seq_ids, split["test_idx"], fasta_path)
    
    tp = sum(1 for i, sid in enumerate(test_ids) if sid in blast_flags and y_test[i] == 1)
    fn = sum(1 for i, sid in enumerate(test_ids) if sid not in blast_flags and y_test[i] == 1)
    blast_tpr = tp / (tp + fn) if (tp + fn) else 0
    blast_fnr = fn / (tp + fn) if (tp + fn) else 0
    
    print(f" BLAST FNR: {blast_fnr:.4f}")
    plot_classifier_vs_blast(clf_results, blast_tpr, blast_fnr)

    # 4. Adversarial
    run_adversarial_experiment(labels, seq_ids, split, best_pos, lr_clf, mlp_clf, scaler, fasta_path)

    print(f"\n Pipeline Complete. Artifacts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
