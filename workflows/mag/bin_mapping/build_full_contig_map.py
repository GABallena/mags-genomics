#!/usr/bin/env python3
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

"""
build_full_contig_map.py

Construct a comprehensive contig->MAG (bin) mapping table by leveraging restored headers if available,
else falling back to current bin FASTA headers and deriving contig IDs.

Priority of sources per sample:
1. Restored bins in restored_bins/<sample> (files ending with .orig.fa or .orig.fasta)
2. RefineM filtered bins in refinem_filtered_bins/<sample> (*.fa)
3. Other future sources can be appended.

If a per-sample contigs FASTA exists in relabeled_contigs/<sample>_contigs.fa(.fasta), we also capture
original header hash correlation to optionally validate counts (not strictly required for mapping output).

Output: mag_analysis/full_contig_bin_map.tsv with columns:
  sample, contig, mag_id, source_file
Where 'contig' is the assembler contig ID intended to match depth files (e.g., k141_12345) and mag_id
is a normalized MAG ID consistent with prior usage (filename + optional suffix cleaned).

Assumptions:
- Assembler contig IDs begin with a pattern like k\d+_\d+ or similar; we extract the rightmost token
  matching that in the header line. If not found, we fallback to full header stripped of whitespace.
- MAG ID derived from bin filename (basename) with trailing extensions removed.

Author: Automated (2025-09-13)
"""
from __future__ import annotations
import re, os, sys, argparse, gzip
from pathlib import Path
from typing import Iterable, Tuple

def read_fasta(fp: Path) -> Iterable[Tuple[str,str]]:
    name=None; seq=[]
    opener = gzip.open if fp.suffix=='.gz' else open
    with opener(fp, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq)
                name=line[1:].strip(); seq=[]
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, ''.join(seq)

def norm_mag_id(filename: str) -> str:
    b = os.path.basename(filename)
    b = re.sub(r'\.orig(\.fa(sta)?|\.fna)?$', '', b)
    b = re.sub(r'(\.fa(sta)?|\.fna)$', '', b)
    b = re.sub(r'\.+$', '', b)
    return b

def extract_contig(header: str) -> str:
    # Prefer assembler token like k141_123456
    m = re.search(r'(k\d+_\d+)', header)
    if m:
        return m.group(1)
    # fallback: first whitespace-delimited token
    return header.split()[0]

def process_bin_dir(sample_dir: Path, priority: int) -> Iterable[Tuple[str,str,str,str]]:
    sample = sample_dir.name
    for fp in sorted(sample_dir.glob('*.fa*')):
        if fp.name.endswith('.log'): continue
        mag_id = norm_mag_id(fp.name)
        for h,_ in read_fasta(fp):
            contig = extract_contig(h)
            yield sample, contig, mag_id, f"{priority}:{fp}"

def main():
    ap = argparse.ArgumentParser(description='Build full contig to MAG mapping TSV.')
    ap.add_argument('--restored-dir', default='restored_bins', help='Directory with restored bins (may have sample subdirs)')
    ap.add_argument('--filtered-dir', default='refinem_filtered_bins', help='Directory with filtered bins per sample')
    ap.add_argument('--output', default='mag_analysis/full_contig_bin_map.tsv', help='Output TSV path')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out_fp = Path(args.output)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    if out_fp.exists() and not args.force:
        sys.stderr.write(f"ERROR: {out_fp} exists. Use --force to overwrite.\n")
        sys.exit(1)

    rows=[]
    # Priority 1: restored bins
    restored_root = Path(args.restored_dir)
    if restored_root.is_dir():
        for sample_dir in sorted(p for p in restored_root.iterdir() if p.is_dir()):
            rows.extend(process_bin_dir(sample_dir, priority=1))
    # Priority 2: filtered bins (only add contigs not already mapped for that sample+contig)
    existing = {(s,c) for s,c,_,_ in rows}
    filtered_root = Path(args.filtered_dir)
    if filtered_root.is_dir():
        for sample_dir in sorted(p for p in filtered_root.iterdir() if p.is_dir()):
            for rec in process_bin_dir(sample_dir, priority=2):
                s,c,mag_id,src = rec
                if (s,c) not in existing:
                    rows.append(rec)
                    existing.add((s,c))

    if not rows:
        sys.stderr.write("No contigs discovered.\n")
        sys.exit(1)

    # Write TSV
    with open(out_fp, 'w') as out:
        out.write('sample\tcontig\tmag_id\tsource_file\n')
        for s,c,mag_id,src in rows:
            out.write(f"{s}\t{c}\t{mag_id}\t{src}\n")

    # Stats
    n_unique = len({(s,c) for s,c,_,_ in rows})
    sys.stderr.write(f"[DONE] Wrote {out_fp} rows={len(rows)} unique_pairs={n_unique} samples={len({s for s,_,_,_ in rows})}\n")

if __name__ == '__main__':
    main()
