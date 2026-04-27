"""
AIxBio Hackathon 2026 - ESM-2 Activation Extraction
===================================================
Extracts mean-pooled per-layer representations from ESM-2 models for a given 
dataset of protein sequences. Saves the extracted embeddings, labels, and 
metadata to an HDF5 file for fast downstream linear probing.

Features:
  - Supports ESM-2 150M, 650M, and 3B parameter variants.
  - Optionally extracts a subset of checkpoint layers to save VRAM and disk space.
  - Dynamically records physical layer indices for downstream interpretability.

Usage:
  python extract_representations.py --model_size 650M
  python extract_representations.py --model_size 3B --subset_layers --batch_size 4
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

EXTRACT_ALL_LAYERS = True

# If EXTRACT_ALL_LAYERS=False, extract these specific layer indices:
LAYER_INDICES_650M =[0, 4, 8, 11, 16, 22, 27, 32]   # 8 checkpoints across 33 layers
LAYER_INDICES_3B   =[0, 6, 12, 18, 24, 30, 35]      # 7 checkpoints across 36 layers

MAX_SEQ_LEN = 1022   # ESM-2 context window is 1024 with BOS/EOS tokens
BATCH_SIZE  = 8      # Reduce to 4 if encountering OOM errors on smaller GPUs

MODEL_OPTIONS = {
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B":   "facebook/esm2_t36_3B_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------

def load_model(model_size: str = "650M"):
    model_name = MODEL_OPTIONS[model_size]
    print(f"Loading {model_name} ...")

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        add_pooling_layer=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


# -----------------------------------------------------------------------------
# EXTRACTION
# -----------------------------------------------------------------------------

def extract_batch(model, tokenizer, sequences: list[str], device: str) -> np.ndarray:
    """
    Forward pass one batch of sequences.
    Returns mean-pooled representations for each layer:
      shape: [batch_size, num_layers+1, hidden_dim]
      (index 0 = embedding layer, 1..L = transformer layers)
    """
    sequences = [s[:MAX_SEQ_LEN] for s in sequences]

    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN + 2
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    seq_lengths = attention_mask.sum(dim=1)

    layer_reps =[]
    for layer_hidden in hidden_states:
        masked = layer_hidden * attention_mask
        mean_rep = masked.sum(dim=1) / seq_lengths
        layer_reps.append(mean_rep.cpu().float().numpy())

    return np.stack(layer_reps, axis=1)


def extract_all(
    sequences: list[str],
    seq_ids: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = BATCH_SIZE,
    layer_indices: list[int] | None = None
) -> dict:
    """
    Extract representations for all sequences in batches.
    Returns dict: {seq_id: rep_array}
    rep_array shape: [num_layers_extracted, hidden_dim]
    """
    results = {}
    n = len(sequences)

    for i in tqdm(range(0, n, batch_size), desc="Extracting"):
        batch_seqs = sequences[i:i+batch_size]
        batch_ids  = seq_ids[i:i+batch_size]

        reps = extract_batch(model, tokenizer, batch_seqs, device)

        if layer_indices is not None:
            reps = reps[:, layer_indices, :]

        for j, sid in enumerate(batch_ids):
            results[sid] = reps[j]

    return results


# -----------------------------------------------------------------------------
# HDF5 I/O
# -----------------------------------------------------------------------------

def save_hdf5(
    reps: dict,
    labels_df: pd.DataFrame,
    output_path: Path,
    layer_indices: list[int] | None = None
):
    """
    Save extracted representations, labels, and metadata to HDF5.
    Structure:
      /representations/{seq_id}   float32 array [num_layers, hidden_dim]
      /labels                     int32 array   (-1 = seq_id not found in labels_df)
      /seq_ids                    variable-length string array
      /layer_indices              int32 array (physical layer numbers saved)
    """
    seq_ids   = list(reps.keys())
    label_map = dict(zip(labels_df["seq_id"], labels_df["label"]))

    missing = [sid for sid in seq_ids if sid not in label_map]
    if missing:
        print(f"  [WARN] {len(missing)} seq_ids not found in labels.csv -- "
              f"will be stored with label=-1. First 5: {missing[:5]}")
    labels = np.array([label_map.get(sid, -1) for sid in seq_ids], dtype=np.int32)

    if layer_indices is not None:
        saved_layer_indices = np.array(layer_indices, dtype=np.int32)
    else:
        num_layers = next(iter(reps.values())).shape[0]
        saved_layer_indices = np.arange(num_layers, dtype=np.int32)

    with h5py.File(output_path, "w") as f:
        grp = f.create_group("representations")
        for sid, rep in reps.items():
            grp.create_dataset(sid, data=rep.astype(np.float32), compression="gzip")

        f.create_dataset("labels", data=labels)

        # Requires dtype=object for variable length unicode strings in h5py
        dt = h5py.string_dtype()
        f.create_dataset("seq_ids", data=np.array(seq_ids, dtype=object), dtype=dt)
        f.create_dataset("layer_indices", data=saved_layer_indices)

    print(f"Saved {len(reps)} representations -> {output_path}")
    print(f"  layer_indices stored: {saved_layer_indices.tolist()}")


def load_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Load representations from HDF5.
    Returns:
      reps          : [N, num_layers, hidden_dim] float32 array
      labels        : [N] int32 array
      seq_ids       : list of string identifiers
      layer_indices : [num_layers] int32 array (physical ESM-2 layer numbers)
    """
    with h5py.File(path, "r") as f:
        if "layer_indices" not in f:
            raise KeyError(
                "HDF5 file missing /layer_indices. Re-run extraction with the latest script."
            )

        layer_indices = f["layer_indices"][:]
        seq_ids = [s.decode() if isinstance(s, bytes) else s for s in f["seq_ids"][:]]
        labels  = f["labels"][:]

        rep_group = f["representations"]

        missing_keys = [sid for sid in seq_ids if sid not in rep_group]
        if missing_keys:
            raise KeyError(
                f"{len(missing_keys)} seq_ids in /seq_ids have no matching entry in "
                f"/representations. First 3: {missing_keys[:3]}"
            )

        reps = np.stack([rep_group[sid][:] for sid in seq_ids], axis=0)

    return reps, labels, seq_ids, layer_indices


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def run_extraction(
    data_dir: str = "./data",
    model_size: str = "650M",
    batch_size: int = BATCH_SIZE,
    extract_all_layers: bool = EXTRACT_ALL_LAYERS
):
    data_dir = Path(data_dir)
    fasta_path  = data_dir / "sequences.fasta"
    labels_path = data_dir / "labels.csv"
    output_path = data_dir / f"representations_{model_size}.h5"

    assert fasta_path.exists(),  f"Missing: {fasta_path}"
    assert labels_path.exists(), f"Missing: {labels_path}"

    print("Loading sequences ...")
    records   = list(SeqIO.parse(fasta_path, "fasta"))
    sequences = [str(r.seq) for r in records]
    seq_ids   = [r.id for r in records]
    print(f"  {len(sequences)} sequences loaded")

    labels_df = pd.read_csv(labels_path)

    fasta_ids = set(seq_ids)
    csv_ids   = set(labels_df["seq_id"])
    unmatched = fasta_ids - csv_ids
    if unmatched:
        print(f"  [WARN] {len(unmatched)} FASTA seq_ids not in labels.csv -- "
              f"these will get label=-1. First 5: {list(unmatched)[:5]}")

    model, tokenizer, device = load_model(model_size)

    if extract_all_layers:
        layer_indices = None
        print("Extracting ALL layers")
    else:
        layer_indices = LAYER_INDICES_650M if model_size == "650M" else LAYER_INDICES_3B
        print(f"Extracting specific layers: {layer_indices}")

    reps = extract_all(
        sequences, seq_ids, model, tokenizer, device,
        batch_size=batch_size, layer_indices=layer_indices
    )

    save_hdf5(reps, labels_df, output_path, layer_indices=layer_indices)

    sample_id  = seq_ids[0]
    sample_rep = reps[sample_id]
    print(f"\nSanity check -- {sample_id}")
    print(f"  rep shape: {sample_rep.shape}  (expected: [num_layers, hidden_dim])")
    print(f"  num_layers={sample_rep.shape[0]}, hidden_dim={sample_rep.shape[1]}")

    print(f"\nExtraction complete: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-2 Activation Extraction")
    parser.add_argument("--data_dir",      default="./data")
    parser.add_argument("--model_size",    default="650M", choices=["150M", "650M", "3B"])
    parser.add_argument("--batch_size",    default=8, type=int)
    parser.add_argument("--subset_layers", action="store_true", help="Extract only specific checkpoint layers")
    args = parser.parse_args()

    run_extraction(
        data_dir=args.data_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        extract_all_layers=not args.subset_layers
    )
