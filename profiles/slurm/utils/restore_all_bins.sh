#!/usr/bin/env bash
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

set -euo pipefail

RELABEL_DIR="${RELABEL_DIR:-relabeled_contigs}"
BINS_ROOT="${BINS_ROOT:-refinem_filtered_bins}"
OUT_ROOT="${OUT_ROOT:-restored_bins}"
INDEX_DIR="${INDEX_DIR:-restored_index}"

mkdir -p "$OUT_ROOT" "$INDEX_DIR"

# Loop over each sample folder in refinem_filtered_bins
for sample_dir in "$BINS_ROOT"/*; do
    # skip non-directories
    [ -d "$sample_dir" ] || continue

    sample=$(basename "$sample_dir")
    echo "=== Processing sample: $sample ==="

    python3 restore_bin_headers_per_sample.py \
        --sample "$sample" \
        --relabeled-contigs-dir "$RELABEL_DIR" \
        --bins-dir "$BINS_ROOT/$sample" \
        --out-dir "$OUT_ROOT/$sample" \
        --index-dir "$INDEX_DIR"
done

echo "=== All samples done ==="
