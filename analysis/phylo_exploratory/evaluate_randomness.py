#!/usr/bin/env python3
# Portfolio-safe copy: alignment-quality heuristic (randomness)
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path
import numpy as np


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


def column_identity_score(alignment: list[str]) -> int:
    # Count columns where all residues are identical
    score = 0
    for col in zip(*alignment):
        if all(c == col[0] for c in col):
            score += 1
    return score


def random_alignment_score(alignment: list[str], rng: np.random.Generator) -> int:
    randomized = ["".join(rng.permutation(list(seq))) for seq in alignment]
    return column_identity_score(randomized)


def monte_carlo_alignment_test(alignment: list[str], n_simulations: int, seed: int) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    observed = column_identity_score(alignment)
    sims = [random_alignment_score(alignment, rng) for _ in range(n_simulations)]
    p = float(np.mean([s >= observed for s in sims]))
    return observed, p


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo randomness test on an alignment identity score.")
    ap.add_argument("fasta", help="Aligned FASTA file")
    ap.add_argument("--n", type=int, default=1000, help="Number of simulations")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    ap.add_argument("--out", default="results/phylo_quality/randomness_test.txt", help="Output text file")
    args = ap.parse_args()

    aln = load_alignment_fasta(args.fasta)
    observed, p = monte_carlo_alignment_test(aln, args.n, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"ObservedScore\t{observed}\nPValue\t{p}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
