#!/usr/bin/env python3
# Portfolio-safe copy: alignment-quality heuristic (mutual_information)
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import mutual_info_score


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
    # ensure equal length
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("Alignment sequences must have equal length")
    return seqs


def calculate_mutual_information(alignment: list[str]) -> float:
    cols = list(zip(*alignment))
    mi_scores = [
        mutual_info_score(cols[i], cols[j])
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    return float(np.mean(mi_scores)) if mi_scores else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Compute mean pairwise mutual information across alignment columns.")
    ap.add_argument("fasta", help="Aligned FASTA file")
    ap.add_argument("--out", default="results/phylo_quality/mutual_information.txt", help="Output text file")
    args = ap.parse_args()

    aln = load_alignment_fasta(args.fasta)
    mi = calculate_mutual_information(aln)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Mutual Information Score\t{mi}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
