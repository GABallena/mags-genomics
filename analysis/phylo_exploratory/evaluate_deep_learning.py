#!/usr/bin/env python3
# Portfolio-safe copy: alignment-quality heuristic (deep_learning)
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    raise SystemExit("This script requires tensorflow. Install with: pip install tensorflow") from e


RESIDUES = "ACDEFGHIKLMNPQRSTVWY-"
RES_TO_IDX = {r: i for i, r in enumerate(RESIDUES)}


def load_alignment_fasta(file_path: str) -> list[str]:
    seqs = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    if not seqs:
        raise ValueError("No sequences found in FASTA")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("Alignment sequences must have equal length")
    return seqs


def extract_features(alignment: list[str]) -> np.ndarray:
    # Encode each column as integers (shape: n_cols x n_seqs)
    cols = list(zip(*alignment))
    feats = [[RES_TO_IDX.get(res, len(RESIDUES)) for res in col] for col in cols]
    return np.array(feats, dtype=np.int32)


def predict_alignment_quality(alignment: list[str], model_path: str) -> float:
    model = tf.keras.models.load_model(model_path)
    features = extract_features(alignment)
    features = np.expand_dims(features, axis=0)  # batch dim
    pred = model.predict(features, verbose=0)
    return float(pred[0, 0])  # assumes scalar output


def main():
    ap = argparse.ArgumentParser(description="Score an alignment using a pretrained TF model (experimental).")
    ap.add_argument("fasta", help="Aligned FASTA file")
    ap.add_argument("--model", required=True, help="Path to a keras .h5 or SavedModel")
    ap.add_argument("--out", default="results/phylo_quality/deep_learning_score.txt", help="Output text file")
    args = ap.parse_args()

    aln = load_alignment_fasta(args.fasta)
    score = predict_alignment_quality(aln, args.model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"DeepLearningQualityScore\t{score}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
