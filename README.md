# mags-genomics: k-mer signatures and marker workflows

This module groups exploratory, portfolio-safe utilities for genomic QC / feature engineering using k-mers and marker sequences.

## Contents
- `workflows/kmer_signatures/`
  - Snakemake workflows for:
    - marker generation
    - k-mer variance diagnostics
    - k-mer diversity evaluation
    - k-mer contamination checks
- `tools/kmer_signatures/`
  - Python helper scripts used by the workflows.

## Notes
- The scripts are provided as-is, with identifiers generalized for privacy.
- These are intended as reusable building blocks; you will typically adapt paths/inputs to your environment.
