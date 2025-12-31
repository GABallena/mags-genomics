# mags-genomics

Genome-resolved metagenomics utilities for **MAG assembly, QC, taxonomy, and downstream summaries**.

This repo focuses on building a reproducible workflow that turns metagenomic assemblies/bins into:
- quality metrics (completeness/contamination)
- taxonomy labels
- non-redundant MAG sets
- analysis-ready tables for plotting and reporting

## Scope (high level)
- assembly/bin ingestion + metadata harmonization
- QC parsing and thresholding
- dereplication / representative selection
- taxonomy joins + summary tables
- plotting helpers for MAG counts and quality distributions

## Expected inputs
Provide your own inputs locally (do not commit them):
- MAG FASTA files and/or bin sets
- tool outputs (QC, taxonomy, dereplication summaries)
- sample metadata tables (CSV/TSV)

## Quick start
Run scripts from repo root (examples):
```bash
python -m mags_genomics.build_tables --input data/ --out outputs/
python -m mags_genomics.plot --tables outputs/tables/ --out outputs/figures/
```

If your repo is primarily shell/R instead of Python, replace the commands above with your actual entrypoints.


## License
MIT — see `LICENSE`.
