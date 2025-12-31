#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import numpy as np
import sys
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, ks_2samp

# Load fitted distribution results
try:
    fitted_distributions = np.loadtxt(snakemake.input[0])  # Adjust based on your output format
except OSError as e:
    print(f"Error loading fitted distributions: {e}")
    sys.exit(1)

# Check if fitting was unsuccessful
if np.any(fitted_distributions[:, 1] <= 0):  # Example condition for failure
    # Load k-mer counts for non-parametric analysis
    kmer_counts = np.loadtxt("kmer_counts.txt")  # Adjust as necessary
    other_group_data = np.loadtxt("other_group_data.txt")  # Adjust for the second group

    # Perform non-parametric tests
    results = {}

    # Mann-Whitney U Test
    u_statistic, p_value_mw = mannwhitneyu(kmer_counts, other_group_data, alternative='two-sided')
    results['Mann-Whitney U'] = (u_statistic, p_value_mw)

    # Wilcoxon Signed-Rank Test (if data is paired)
    if len(kmer_counts) == len(other_group_data):
        w_statistic, p_value_w = wilcoxon(kmer_counts, other_group_data)
        results['Wilcoxon'] = (w_statistic, p_value_w)

    # Kruskal-Wallis H Test (for more than two groups)
    # Assuming additional data loaded if applicable
    h_statistic, p_value_kw = kruskal(kmer_counts, other_group_data)  # Add more groups if needed
    results['Kruskal-Wallis'] = (h_statistic, p_value_kw)

    # Friedman Test (for related groups, if applicable)
    # Assuming data is structured appropriately
    f_statistic, p_value_f = friedmanchisquare(kmer_counts, other_group_data)  # Add more groups if needed
    results['Friedman'] = (f_statistic, p_value_f)

    # Kolmogorov-Smirnov Test (compares distributions)
    ks_statistic, p_value_ks = ks_2samp(kmer_counts, other_group_data)
    results['Kolmogorov-Smirnov'] = (ks_statistic, p_value_ks)

    # Save results
    with open(snakemake.output[0], 'w') as f:
        for test, (statistic, p_value) in results.items():
            f.write(f"{test}: Statistic = {statistic}, P-value = {p_value}\n")

else:
    print("Fitting was successful; no non-parametric analysis performed.")
    sys.exit(0)  # Or handle accordingly
