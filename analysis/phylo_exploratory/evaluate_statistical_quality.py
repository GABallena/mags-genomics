#!/usr/bin/env python3
# Portfolio-safe copy: alignment-quality heuristic (statistical_quality)
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path
import numpy as np
from scipy.stats import entropy


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


def calculate_conservation(alignment: list[str]) -> float:
    cols = list(zip(*alignment))
    scores = [max(col.count(b) for b in set(col)) / len(col) for col in cols]
    return float(np.mean(scores)) if scores else float("nan")


def calculate_entropy(alignment: list[str]) -> float:
    cols = list(zip(*alignment))
    scores = [
        entropy([col.count(b) / len(col) for b in set(col)], base=2)
        for col in cols
    ]
    return float(np.mean(scores)) if scores else float("nan")


def model_predict(conservation: float, ent: float) -> float:
    # Example heuristic: higher conservation and lower entropy is "better"
    return float(conservation - ent)


def main():
    ap = argparse.ArgumentParser(description="Compute a simple statistical alignment-quality score.")
    ap.add_argument("fasta", help="Aligned FASTA file")
    ap.add_argument("--out", default="results/phylo_quality/stat_quality.txt", help="Output text file")
    args = ap.parse_args()

    aln = load_alignment_fasta(args.fasta)
    cons = calculate_conservation(aln)
    ent = calculate_entropy(aln)
    q = model_predict(cons, ent)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"Conservation\t{cons}\nEntropy\t{ent}\nQualityScore\t{q}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
