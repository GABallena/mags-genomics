#!/usr/bin/env python3
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

"""Compute ARG burden metrics per MAG.

Inputs:
  - Strict ARG mapping table (e.g. mag_analysis/arg_to_mag_strict.tsv) produced by link_args_to_mags.py
  - Depth directory with per-sample contig depth files: depth/<sample>.depth.txt (tsv with header contigName,totalAvgDepth,...)
  - CheckM1 extended stats (bin_stats_ext.tsv) for completeness and genome size.
  - CheckM2 quality report (quality_report.tsv) optional, preferred completeness if present.

Metrics per MAG:
  assembled_length_bp   = genome size (prefer CheckM2 Genome_Size else CheckM1 Genome size (bp))
  completeness_used     = completeness (prefer CheckM2 Completeness else CheckM1 Completeness)
  effective_genome_bp   = assembled_length_bp / (completeness_used/100)
  ARG_orf_count         = unique ARG ORFs mapped to MAG passing coverage filters
  ARG_class_richness    = number of unique drug classes among passing ORFs
  genes_per_Mbp         = ARG_orf_count / (effective_genome_bp / 1e6)

Coverage filter (approximation):
  We only have contig-level mean depth; ORF breadth data is not available. We approximate ORF depth by contig mean depth.
  User-specified thresholds: --min-depth (default 1.0). Breadth filter placeholder: if future per-ORF breadth file provided we can integrate.

Output:
  TSV with columns: mag_id, sample, completeness_chk1, completeness_chk2, completeness_used, genome_size_bp, effective_genome_bp, ARG_orf_count, ARG_class_richness, genes_per_Mbp, depth_filtered_orfs, total_orfs_mag

Limitations:
  - ORF breadth >= 0.5 not enforced due to missing per-ORF coverage profile; contig-level average depth used instead.
  - MAG IDs are normalized from bin FASTA filenames by stripping trailing '.orig.fa'.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
from collections import defaultdict


def read_checkm1(path: str):
    data = {}
    if not path or not os.path.isfile(path):
        return data
    with open(path, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = None
        for row in reader:
            if not row or row[0].startswith('['):
                continue
            if row[0] == 'Bin Id':
                header = row
                # Identify columns
                continue
            if header is None:
                continue
            # Expected columns include 'Bin Id', 'Completeness', 'Genome size (bp)'
            try:
                bin_id = row[0].strip()
                completeness = float(row[5])  # 'Completeness'
                genome_size = int(row[8])      # 'Genome size (bp)'
            except (ValueError, IndexError):
                continue
            data[bin_id] = {
                'completeness_chk1': completeness,
                'genome_size_chk1': genome_size,
            }
    return data


def read_checkm2(path: str):
    data = {}
    if not path or not os.path.isfile(path):
        return data
    with open(path, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = next(reader, None)
        if not header:
            return data
        # Columns: Name, Completeness, Contamination, ..., Genome_Size
        try:
            name_idx = header.index('Name')
            comp_idx = header.index('Completeness')
            gsize_idx = header.index('Genome_Size')
        except ValueError:
            return data
        for row in reader:
            if not row:
                continue
            try:
                name = row[name_idx].strip()
                completeness = float(row[comp_idx])
                genome_size = int(float(row[gsize_idx]))
            except (ValueError, IndexError):
                continue
            data[name] = {
                'completeness_chk2': completeness,
                'genome_size_chk2': genome_size,
            }
    return data


def load_depth_dir(depth_dir: str, min_depth: float):
    """Return mapping: sample -> contig_short -> depth (mean)."""
    depth = {}
    if not depth_dir or not os.path.isdir(depth_dir):
        return depth
    for fname in os.listdir(depth_dir):
        if not fname.endswith('.depth.txt'):
            continue
        sample = fname.split('.depth.txt')[0]
        fp = os.path.join(depth_dir, fname)
        contig_cov = {}
        with open(fp, 'r') as fh:
            reader = csv.reader(fh, delimiter='\t')
            header = next(reader, None)
            if not header:
                continue
            # totalAvgDepth assumed at index 2 if header matches example
            for row in reader:
                if len(row) < 3:
                    continue
                contig = row[0]
                try:
                    cov = float(row[2])
                except ValueError:
                    continue
                contig_cov[contig] = cov
        depth[sample] = contig_cov
    return depth


def normalize_mag_id(bin_file: str):
    # Strip .orig.fa and any leading path
    base = os.path.basename(bin_file)
    if base.endswith('.orig.fa'):
        base = base[:-len('.orig.fa')]
    return base


def parse_args_table(arg_tsv: str, depth_map, min_depth: float):
    """Return per-mag aggregated ORF info after depth filter.

    Returns dict mag_id -> {'sample': sample, 'orf_ids': set, 'drug_classes': set, 'total_orfs': int, 'depth_pass_orfs': int}
    """
    agg = {}
    with open(arg_tsv, 'r') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            bin_file = row.get('bin_file') or ''
            if bin_file in ('NA', '', None):
                continue  # skip unbinned
            sample = row.get('sample')
            contig_base = row.get('contig_base') or ''
            orf_id = row.get('orf_id')
            drug_classes = row.get('drug_classes') or ''
            mag_id = normalize_mag_id(bin_file)
            if mag_id not in agg:
                agg[mag_id] = {
                    'sample': sample,
                    'orf_ids': set(),
                    'drug_classes': set(),
                    'total_orfs': 0,
                    'depth_pass_orfs': 0,
                }
            entry = agg[mag_id]
            entry['total_orfs'] += 1
            # Coverage lookup: contig_base pattern sample + '_' + contig
            contig_short = contig_base
            if sample and contig_base.startswith(sample + '_'):
                contig_short = contig_base[len(sample)+1:]
            cov = depth_map.get(sample, {}).get(contig_short)
            depth_pass = cov is not None and cov >= min_depth
            if depth_pass:
                entry['depth_pass_orfs'] += 1
                entry['orf_ids'].add(orf_id)
                # split drug classes by semicolon
                for tok in re.split(r';\s*', drug_classes):
                    t = tok.strip()
                    if t:
                        entry['drug_classes'].add(t)
    return agg


def main():
    ap = argparse.ArgumentParser(description="Compute ARG burden metrics per MAG")
    ap.add_argument('--arg-tsv', default='mag_analysis/arg_to_mag_strict.tsv', help='Strict ARG to MAG mapping TSV')
    ap.add_argument('--depth-dir', default='depth', help='Directory of per-sample contig depth files')
    ap.add_argument('--checkm1', default='checkm1_out/bin_stats_ext.tsv', help='CheckM1 bin_stats_ext.tsv path')
    ap.add_argument('--checkm2', default='checkm2_out/quality_report.tsv', help='CheckM2 quality_report.tsv path')
    ap.add_argument('--min-depth', type=float, default=1.0, help='Minimum mean contig depth to retain ORF (proxy for ORF depth)')
    ap.add_argument('-o', '--output', default='mag_analysis/arg_burden.tsv', help='Output TSV path')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    checkm1 = read_checkm1(args.checkm1)
    checkm2 = read_checkm2(args.checkm2)
    depth_map = load_depth_dir(args.depth_dir, args.min_depth)
    agg = parse_args_table(args.arg_tsv, depth_map, args.min_depth)

    # Merge metrics and write
    fieldnames = [
        'mag_id', 'sample', 'completeness_chk1', 'completeness_chk2', 'completeness_used',
        'genome_size_bp', 'effective_genome_bp', 'ARG_orf_count', 'ARG_class_richness',
        'genes_per_Mbp', 'depth_filtered_orfs', 'total_orfs_mag'
    ]
    with open(args.output, 'w', newline='') as out:
        w = csv.writer(out, delimiter='\t')
        w.writerow(fieldnames)
        for mag_id, info in sorted(agg.items()):
            c1 = c2 = g1 = g2 = None
            if mag_id in checkm1:
                c1 = checkm1[mag_id]['completeness_chk1']
                g1 = checkm1[mag_id]['genome_size_chk1']
            if mag_id in checkm2:
                c2 = checkm2[mag_id]['completeness_chk2']
                g2 = checkm2[mag_id]['genome_size_chk2']
            completeness_used = c2 if c2 is not None else c1
            genome_size_bp = g2 if g2 is not None else g1
            effective_genome_bp = None
            genes_per_Mbp = None
            if completeness_used and genome_size_bp and completeness_used > 0:
                effective_genome_bp = genome_size_bp / (completeness_used / 100.0)
            arg_orf_count = len(info['orf_ids'])
            arg_class_richness = len(info['drug_classes'])
            if effective_genome_bp and effective_genome_bp > 0:
                genes_per_Mbp = arg_orf_count / (effective_genome_bp / 1e6)
            w.writerow([
                mag_id,
                info['sample'],
                f"{c1:.2f}" if c1 is not None else 'NA',
                f"{c2:.2f}" if c2 is not None else 'NA',
                f"{completeness_used:.2f}" if completeness_used is not None else 'NA',
                genome_size_bp if genome_size_bp is not None else 'NA',
                f"{effective_genome_bp:.2f}" if effective_genome_bp is not None else 'NA',
                arg_orf_count,
                arg_class_richness,
                f"{genes_per_Mbp:.4f}" if genes_per_Mbp is not None else 'NA',
                info['depth_pass_orfs'],
                info['total_orfs']
            ])

    print(f"Wrote burden metrics for {len(agg)} MAGs to {args.output}")
    missing_meta = [m for m in agg if m not in checkm1 and m not in checkm2]
    if missing_meta:
        print(f"Warning: {len(missing_meta)} MAGs lacked completeness/genome size (first 5): {missing_meta[:5]}")


if __name__ == '__main__':
    main()
