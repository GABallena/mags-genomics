#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm
import sys

# Load k-mer counts from the input file
try:
    kmer_counts = np.loadtxt(snakemake.input[0])
except OSError as e:
    print(f"Error loading input file: {e}")
    sys.exit(1)

# Plot a histogram of k-mer counts
plt.hist(kmer_counts, bins=50, color='blue', alpha=0.7)
plt.title('K-mer Count Distribution')
plt.xlabel('K-mer Counts')
plt.ylabel('Frequency')
plt.savefig(snakemake.output[0])  # Save histogram as an output file
plt.close()

# Perform Shapiro-Wilk normality test
stat, p_value = shapiro(kmer_counts)
print(f'Shapiro-Wilk test: Statistic={stat}, P-value={p_value}')

# Perform Kolmogorov-Smirnov test for comparison
ks_stat, ks_p_value = kstest(kmer_counts, 'norm', args=(np.mean(kmer_counts), np.std(kmer_counts)))
print(f'Kolmogorov-Smirnov test: Statistic={ks_stat}, P-value={ks_p_value}')

# Write the test results to a file
with open(snakemake.output[1], 'w') as f:
    f.write(f"Shapiro-Wilk test: Statistic={stat}, P-value={p_value}\n")
    f.write(f"Kolmogorov-Smirnov test: Statistic={ks_stat}, P-value={ks_p_value}\n")

# Decide if data is normal based on p-value
if p_value < 0.05:
    f.write("Data is not normally distributed based on Shapiro-Wilk test.\n")
else:
    f.write("Data appears normally distributed based on Shapiro-Wilk test.\n")

if ks_p_value < 0.05:
    f.write("Data is not normally distributed based on Kolmogorov-Smirnov test.\n")
else:
    f.write("Data appears normally distributed based on Kolmogorov-Smirnov test.\n")
