#!/usr/bin/env python3
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

import argparse, gzip, hashlib, os, sys
from pathlib import Path

def hash_seq(seq: str) -> str:
    return hashlib.sha1(''.join(seq.split()).upper().encode('utf-8')).hexdigest()

def read_fasta(fp):
    h=None; buf=[]
    with open(fp,'r') as f:
        for line in f:
            if line.startswith('>'):
                if h is not None: yield h, ''.join(buf)
                h=line[1:].strip(); buf=[]
            else:
                buf.append(line.strip())
        if h is not None: yield h, ''.join(buf)

def write_fasta(records, out_fp, width=80):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp,'w') as out:
        for h, s in records:
            out.write(f">{h}\n")
            for i in range(0, len(s), width):
                out.write(s[i:i+width] + "\n")

def build_index_tsv(contigs_fp: Path, idx_fp: Path):
    print(f"[INDEX] Building {idx_fp} from {contigs_fp}")
    idx_fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(idx_fp, 'wt') as gz:
        n=0
        for h,s in read_fasta(contigs_fp):
            gz.write(f"{hash_seq(s)}\t{h}\n"); n+=1
    print(f"[INDEX] Wrote {n} rows")

def load_index_tsv(idx_fp: Path):
    print(f"[INDEX] Loading {idx_fp}")
    m = {}
    with gzip.open(idx_fp, 'rt') as gz:
        for line in gz:
            sha, h = line.rstrip('\n').split('\t', 1)
            # keep first seen header; duplicates collapse
            if sha not in m: m[sha] = h
    print(f"[INDEX] Loaded {len(m)} unique hashes")
    return m

def main():
    ap = argparse.ArgumentParser(description="Restore original headers for ONE sample (per-sample, on-disk index).")
    ap.add_argument("--sample", required=True, help="Sample ID, e.g., SAMPLE-XX_SYY")
    ap.add_argument("--relabeled-contigs-dir", required=True)
    ap.add_argument("--bins-dir", required=True, help="Dir with this sample's bins (e.g., refinem_filtered_bins/SAMPLE-XX_SYY)")
    ap.add_argument("--out-dir", required=True, help="Output dir for restored bins of this sample")
    ap.add_argument("--index-dir", default="restored_index", help="Where to store per-sample gz-index files")
    ap.add_argument("--contigs-suffix", default="_contigs.fa", help="Suffix of contigs FASTA")
    ap.add_argument("--bin-ext", default=".fa", help="Process bin files ending with this")
    args = ap.parse_args()

    sample = args.sample
    contigs_fp = Path(args.relabeled_contigs_dir) / f"{sample}{args.contigs_suffix}"
    if not contigs_fp.exists():
        # also try .fasta
        alt = Path(args.relabeled_contigs_dir) / f"{sample}{args.contigs_suffix.replace('.fa','.fasta')}"
        if alt.exists(): contigs_fp = alt
        else: sys.exit(f"[ERR] Contigs FASTA not found for {sample}: {contigs_fp}")

    bins_dir = Path(args.bins_dir)
    if not bins_dir.is_dir():
        sys.exit(f"[ERR] Bins dir not found: {bins_dir}")

    out_dir = Path(args.out_dir)
    idx_fp = Path(args.index_dir) / f"{sample}.sha1.tsv.gz"

    # build or load index
    if not idx_fp.exists():
        build_index_tsv(contigs_fp, idx_fp)
    m = load_index_tsv(idx_fp)

    # process bins
    bin_files = sorted(p for p in bins_dir.glob(f"*{args.bin_ext}") if p.is_file())
    if not bin_files:
        print(f"[WARN] No bin files with ext {args.bin_ext} in {bins_dir}")
        return

    total=0; unmatched_total=0
    for bin_fp in bin_files:
        restored=[]
        unmatched=0; n=0
        for h,s in read_fasta(bin_fp):
            n+=1; sha=hash_seq(s)
            if sha in m:
                restored.append((m[sha], s))
            else:
                unmatched+=1
                restored.append((f"{h} |UNMATCHED", s))
        out_fp = out_dir / bin_fp.name.replace(args.bin_ext, f".orig{args.bin_ext}")
        write_fasta(restored, out_fp)
        total += n; unmatched_total += unmatched
        print(f"[OK] {bin_fp.name} -> {out_fp.name} | contigs: {n} | unmatched: {unmatched}")

    print(f"[SUMMARY] sample={sample} bins={len(bin_files)} contigs={total} unmatched={unmatched_total}")

if __name__ == "__main__":
    main()
