#!/usr/bin/env python3
# Portfolio-safe copy: wavelet-based normalization (scgs)
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path

import numpy as np
import pywt
from scipy.stats import zscore


def wavelet_normalize(x: np.ndarray, wavelet_name: str = "morl", wmin: int = 1, wmax: int = 20, z_thresh: float = 1.96) -> np.ndarray:
    widths = np.arange(wmin, wmax)
    coeffs, _freq = pywt.cwt(x, widths, wavelet_name)
    z = zscore(coeffs, axis=0)
    significant = np.where(np.abs(z) >= z_thresh, coeffs, 0.0)
    return np.mean(significant, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Wavelet-based normalization with Z-score filtering.")
    ap.add_argument("input", help="Input vector (one column of counts)")
    ap.add_argument("--out", default="results/wavelet/normalized_scgs.txt", help="Output file")
    ap.add_argument("--wavelet", default="morl", help="CWT wavelet name")
    ap.add_argument("--wmin", type=int, default=1, help="Min width (inclusive)")
    ap.add_argument("--wmax", type=int, default=20, help="Max width (exclusive)")
    ap.add_argument("--z", type=float, default=1.96, help="Z-score threshold (two-tailed ~0.05 => 1.96)")
    args = ap.parse_args()

    x = np.loadtxt(args.input)
    y = wavelet_normalize(x, wavelet_name=args.wavelet, wmin=args.wmin, wmax=args.wmax, z_thresh=args.z)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, y)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
