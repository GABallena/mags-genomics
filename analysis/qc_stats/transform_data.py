#!/usr/bin/env python3
# Portfolio-safe copy: transform 1D count vectors based on a selected distribution label.
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path

import numpy as np


def transform(x: np.ndarray, dist_name: str) -> np.ndarray:
    dist = dist_name.lower()
    if dist in {"lognormal", "lognorm"}:
        return np.log(x + 1.0)
    if dist in {"exponential", "expon"}:
        m = float(np.mean(x)) if float(np.mean(x)) != 0.0 else 1.0
        return x / m
    if dist in {"gamma"}:
        m = float(np.mean(x)) if float(np.mean(x)) != 0.0 else 1.0
        return (x / m) ** 2
    return x


def main():
    ap = argparse.ArgumentParser(description="Transform a 1D count vector based on the best-fit distribution label.")
    ap.add_argument("counts", help="Input vector file (one column counts)")
    ap.add_argument("--dist", required=True, help="Best-fit distribution label (e.g., lognormal, exponential, gamma)")
    ap.add_argument("--out", default="results/distribution_fit/transformed_counts.txt", help="Output file")
    args = ap.parse_args()

    x = np.loadtxt(args.counts)
    y = transform(x, args.dist)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, y)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
