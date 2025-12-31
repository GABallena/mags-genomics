#!/usr/bin/env python3
# Portfolio-safe copy: build k-mer matrix from sequences
# Identifiers generalized; no private data included.

import pandas as pd
from collections import defaultdict

# Function to load k-mers and their counts from a file
def load_kmers(kmer_file):
    kmer_counts = defaultdict(dict)
    with open(kmer_file, 'r') as f:
        for line in f:
            kmer, count = line.strip().split()  # Assuming kmer and count are space-separated
            scg = "determine_scg_based_on_kmer"  # You may need to implement SCG identification logic
            kmer_counts[scg][kmer] = int(count)
    return kmer_counts

# Load k-mers from the file
kmer_counts = load_kmers("results/scg_kmers.fa")

# Create a DataFrame (k-mer matrix) where rows are SCGs and columns are k-mer counts
df = pd.DataFrame.from_dict(kmer_counts, orient='index', fill_value=0)

# Save the k-mer matrix to a CSV file
df.to_csv("results/kmer_matrix.csv")
