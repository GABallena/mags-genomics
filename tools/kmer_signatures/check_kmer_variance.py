#!/usr/bin/env python3
# Portfolio-safe copy: diagnostics for k-mer variance
# Identifiers generalized; no private data included.

import pandas as pd

# Load the k-mer count matrix from the CSV file
df = pd.read_csv("results/kmer_matrix.csv", index_col=0)

# Calculate variance of k-mer counts across SCGs
variances = df.var(axis=1)

# Save the variance results to a text file
variances.to_csv("results/variance_per_scg.txt", header=True)
