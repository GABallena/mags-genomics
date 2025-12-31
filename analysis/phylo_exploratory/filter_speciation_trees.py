#!/usr/bin/env python3
# Portfolio-safe copy: filter items based on inferred-event labels and a significance file.
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path


def parse_events_tsv(events_file: str, event_name: str) -> set[str]:
    # expects: Node\tEvent\t...
    keep = set()
    with open(events_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            node, ev = parts[0], parts[1]
            if ev.lower() == event_name.lower():
                keep.add(node)
    return keep


def parse_pvalue_list(pvalues_file: str, threshold: float) -> set[str]:
    # expects: item\tPValue  OR  item: pvalue
    keep = set()
    with open(pvalues_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                item, pv = line.split("\t", 1)
            elif ": " in line:
                item, pv = line.split(": ", 1)
            else:
                continue
            try:
                if float(pv) < threshold:
                    keep.add(item)
            except ValueError:
                continue
    return keep


def main():
    ap = argparse.ArgumentParser(description="Intersect event-labeled nodes with significant items.")
    ap.add_argument("events_tsv", help="Events TSV from infer_events.py")
    ap.add_argument("pvalues_file", help="P-values list (item\tPValue or item: PValue)")
    ap.add_argument("--event", default="Speciation", help="Event label to keep")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    ap.add_argument("--out", default="results/phylo_events/speciation_significant.txt", help="Output list path")
    args = ap.parse_args()

    a = parse_events_tsv(args.events_tsv, args.event)
    b = parse_pvalue_list(args.pvalues_file, args.alpha)
    valid = sorted(a & b)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
