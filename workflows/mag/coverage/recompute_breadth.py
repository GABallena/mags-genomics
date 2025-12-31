#!/usr/bin/env python3
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

"""
recompute_breadth.py

Estimate per-contig breadth (fraction of bases with >=1x coverage) from MetaBAT / jgi_summarize_bam_contig_depths style depth summary files
when per-position depth is NOT available.

INPUT FORMAT ASSUMED (one file per sample in a directory, like depth/SAMPLE-XX_SYY.depth.txt):
Header (tab): contigName  contigLen  totalAvgDepth  <sample>.bam  <sample>.bam-var
Each subsequent row: values for that contig.
Typically the 3rd and 4th columns (totalAvgDepth and <sample>.bam) are identical; we take column 4 as mean depth and column 5 as per-base depth variance.

BREADTH ESTIMATION:
True breadth requires knowing how many positions have depth >=1. Lacking per-position data, we approximate the probability a base has zero coverage (P0)
using two models:
  Poisson:        P0_pois = exp(-m)
  Negative Binom: Given mean m and variance v > m, r = m^2 / (v - m); P0_nb = (r / (r + m))^r
Estimated breadth >=1x:  breadth = 1 - P0
We output both poisson_breadth and nb_breadth, and choose best_breadth = nb_breadth if v > m (over-dispersion), else poisson_breadth.

OUTPUT:
TSV (default: mag_analysis/contig_breadth_estimates.tsv) with columns:
 sample, contig, length_bp, mean_depth, var_depth, poisson_breadth, nb_breadth, best_breadth,
 covered_bp_est, depth_ge1 (bool), breadth_ge_thresh (bool), pass_gate (bool)

GATING (for convenience):
--min-depth (default 1) and --min-breadth (default 0.5) produce boolean flags (depth_ge1, breadth_ge_thresh, pass_gate) but do not filter rows unless --filter is passed.
If --filter is set, only rows passing both thresholds are written.

CAVEATS:
- Approximations; with chimeric / uneven coverage the NB model may still mis-estimate breadth.
- For very low depth (<0.1) both models converge: breadth ≈ depth.
- For very high depths (>50x) breadth ~1 quickly; estimation error negligible.

Author: Automated helper (2025-09-13)
"""
from __future__ import annotations
import argparse, os, sys, math, gzip
from typing import List


def parse_args():
    ap = argparse.ArgumentParser(description="Estimate contig breadth from per-contig mean & variance depth summaries.")
    ap.add_argument('--depth-dir', default='depth', help='Directory containing per-sample *.depth.txt files')
    ap.add_argument('--output', default='mag_analysis/contig_breadth_estimates.tsv', help='Output TSV path')
    ap.add_argument('--min-depth', type=float, default=1.0, help='Depth threshold (>=) for gate flags')
    ap.add_argument('--min-breadth', type=float, default=0.5, help='Breadth threshold (>=) for gate flags')
    ap.add_argument('--filter', action='store_true', help='Write only rows that pass both gates')
    ap.add_argument('--force', action='store_true', help='Overwrite existing output')
    return ap.parse_args()


def open_maybe_gzip(path: str):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def detect_sample_from_filename(path: str) -> str:
    base = os.path.basename(path)
    for suf in ('.depth.txt', '.depth.tsv', '.depth', '.txt', '.tsv'):
        if base.endswith(suf):
            return base[:-len(suf)]
    return os.path.splitext(base)[0]


def estimate_breadths(mean_depth: float, var_depth: float):
    # Poisson
    if mean_depth < 0: mean_depth = 0.0
    try:
        p0_pois = math.exp(-mean_depth)
    except OverflowError:
        p0_pois = 0.0 if mean_depth > 0 else 1.0
    poisson_breadth = 1 - p0_pois
    # Negative binomial only if overdispersed (variance > mean)
    if var_depth is not None and var_depth > mean_depth and mean_depth > 0:
        denom = (var_depth - mean_depth)
        if denom <= 0:
            nb_breadth = poisson_breadth
        else:
            r = (mean_depth * mean_depth) / denom
            # clamp r
            if r <= 0:
                nb_breadth = poisson_breadth
            else:
                p = r / (r + mean_depth)  # success prob
                # P0 = p^r
                try:
                    p0_nb = math.exp(r * math.log(p))
                except ValueError:
                    p0_nb = p0_pois
                nb_breadth = 1 - p0_nb
    else:
        nb_breadth = poisson_breadth
    best = nb_breadth if var_depth is not None and var_depth > mean_depth else poisson_breadth
    return poisson_breadth, nb_breadth, best


def parse_depth_file(path: str, min_depth: float, min_breadth: float):
    rows = []
    sample = detect_sample_from_filename(path)
    with open_maybe_gzip(path) as fh:
        header = fh.readline().strip().split('\t')
        if len(header) < 5 or not header[0].lower().startswith('contig'):
            sys.stderr.write(f"[WARN] Unexpected header in {path}: {' | '.join(header[:6])}\n")
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            contig = parts[0]
            try:
                length = int(parts[1])
                # We take column 3 or 4? Use column 4 (mean for this sample) and column 5 (variance)
                mean_depth = float(parts[3])
                var_depth = float(parts[4])
            except ValueError:
                continue
            poisson_breadth, nb_breadth, best_breadth = estimate_breadths(mean_depth, var_depth)
            covered_bp_est = best_breadth * length
            depth_ge1 = mean_depth >= min_depth  # Approx; true gate ideally at per-position min-depth logic
            breadth_ge = best_breadth >= min_breadth
            pass_gate = depth_ge1 and breadth_ge
            rows.append({
                'sample': sample,
                'contig': contig,
                'length_bp': length,
                'mean_depth': round(mean_depth, 5),
                'var_depth': round(var_depth, 5),
                'poisson_breadth': round(poisson_breadth, 5),
                'nb_breadth': round(nb_breadth, 5),
                'best_breadth': round(best_breadth, 5),
                'covered_bp_est': round(covered_bp_est, 2),
                'depth_ge_threshold': int(depth_ge1),
                'breadth_ge_threshold': int(breadth_ge),
                'pass_gate': int(pass_gate),
            })
    return rows


def main():
    args = parse_args()
    if os.path.exists(args.output) and not args.force:
        sys.stderr.write(f"ERROR: Output exists ({args.output}); use --force to overwrite.\n")
        sys.exit(1)
    if not os.path.isdir(args.depth_dir):
        sys.stderr.write(f"ERROR: depth directory not found: {args.depth_dir}\n")
        sys.exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    depth_files = [os.path.join(args.depth_dir, f) for f in os.listdir(args.depth_dir) if f.endswith('.depth.txt') or f.endswith('.depth.tsv') or f.endswith('.depth')]
    depth_files = sorted(depth_files)
    if not depth_files:
        sys.stderr.write("ERROR: No compatible depth files found in directory.\n")
        sys.exit(1)

    all_rows: List[dict] = []
    for fp in depth_files:
        all_rows.extend(parse_depth_file(fp, args.min_depth, args.min_breadth))

    if args.filter:
        all_rows = [r for r in all_rows if r['pass_gate'] == 1]

    if not all_rows:
        sys.stderr.write("WARNING: No rows after processing (maybe all filtered?).\n")

    # Write TSV
    cols = list(all_rows[0].keys()) if all_rows else ['sample','contig','length_bp','mean_depth','var_depth','poisson_breadth','nb_breadth','best_breadth','covered_bp_est','depth_ge_threshold','breadth_ge_threshold','pass_gate']
    with open(args.output, 'w') as out:
        out.write('\t'.join(cols) + '\n')
        for r in all_rows:
            out.write('\t'.join(str(r[c]) for c in cols) + '\n')
    sys.stderr.write(f"[DONE] wrote {args.output} with {len(all_rows)} rows from {len(depth_files)} samples.\n")
    sys.stderr.write("Columns: " + ', '.join(cols) + '\n')
    sys.stderr.write("NOTE: breadth estimates are approximations (see script header).\n")

if __name__ == '__main__':
    main()
