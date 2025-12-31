#!/usr/bin/env python3
# Portfolio-safe copy: infer duplication/speciation events via a simple species-overlap heuristic.
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path

try:
    from ete3 import Tree
except Exception as e:
    raise SystemExit("This script requires ete3. Install with: pip install ete3") from e


def species_of(leaf_name: str, delim: str) -> str:
    # default: species token is before the first delimiter (e.g., Species_gene)
    return leaf_name.split(delim, 1)[0] if delim in leaf_name else leaf_name


def infer_events(tree_file: str, output_file: str, delim: str = "_") -> None:
    tree = Tree(tree_file, format=1)
    rows = []

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            continue
        if len(node.children) != 2:
            # skip non-bifurcating nodes for this simple heuristic
            continue

        left = set(species_of(leaf.name, delim) for leaf in node.children[0].get_leaves())
        right = set(species_of(leaf.name, delim) for leaf in node.children[1].get_leaves())
        event = "Duplication" if (left & right) else "Speciation"

        rows.append((node.name or "internal_node", event, ",".join(sorted(left)), ",".join(sorted(right))))

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("Node\tEvent\tLeft_Species\tRight_Species\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Infer duplication/speciation labels for internal nodes (species overlap).")
    ap.add_argument("tree", help="Input Newick tree")
    ap.add_argument("--out", default="results/phylo_events/inferred_events.tsv", help="Output TSV path")
    ap.add_argument("--delim", default="_", help="Delimiter separating species token from gene id")
    args = ap.parse_args()

    infer_events(args.tree, args.out, delim=args.delim)


if __name__ == "__main__":
    main()
