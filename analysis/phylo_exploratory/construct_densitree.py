#!/usr/bin/env python3
# Portfolio-safe copy: simple DensiTree-style overlay using ETE3.
# Identifiers generalized; no private data included.

import argparse
from pathlib import Path

try:
    from ete3 import Tree, TreeStyle, NodeStyle, faces
except Exception as e:
    raise SystemExit("This script requires ete3. Install with: pip install ete3") from e


def construct_densitree(tree_files, output_image, title="DensiTree overlay"):
    trees = [Tree(str(p)) for p in tree_files]

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.show_scale = False
    ts.title.add_face(faces.TextFace(title, fsize=18), column=0)

    # Style all trees similarly (thin lines) then render first tree
    for t in trees:
        for node in t.traverse():
            nstyle = NodeStyle()
            nstyle["hz_line_width"] = 1
            nstyle["vt_line_width"] = 1
            node.set_style(nstyle)

    out = Path(output_image)
    out.parent.mkdir(parents=True, exist_ok=True)
    trees[0].render(str(out), tree_style=ts)


def main():
    ap = argparse.ArgumentParser(description="Render an overlay of multiple Newick trees (DensiTree-style).")
    ap.add_argument("trees", nargs="+", help="Input Newick tree files")
    ap.add_argument("--out", default="results/densitree/densitree.png", help="Output image path (PNG/SVG/PDF)")
    ap.add_argument("--title", default="DensiTree overlay", help="Plot title")
    args = ap.parse_args()

    construct_densitree([Path(t) for t in args.trees], args.out, title=args.title)


if __name__ == "__main__":
    main()
