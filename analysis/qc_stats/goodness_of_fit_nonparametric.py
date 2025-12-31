#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import numpy as np
from scipy import stats
import sys

# Load k-mer counts
try:
    kmer_counts = np.loadtxt(snakemake.input[0])
except OSError as e:
    print(f"Error loading input file: {e}")
    sys.exit(1)

# Function to perform goodness-of-fit tests
def goodness_of_fit_tests(data):
    results = {}

    # Kolmogorov-Smirnov Test for Uniform Distribution
    ks_stat, ks_p_value = stats.kstest(data, 'uniform')
    results['KS Test (Uniform)'] = {'Statistic': ks_stat, 'P-value': ks_p_value}

    # Anderson-Darling Test for Uniform Distribution
    ad_stat, critical_values, sig_level = stats.anderson(data, dist='uniform')
    results['Anderson-Darling Test (Uniform)'] = {'Statistic': ad_stat, 'Critical Values': critical_values}

    # Kolmogorov-Smirnov Test for Exponential Distribution
    ks_stat_exp, ks_p_value_exp = stats.kstest(data, 'expon')
    results['KS Test (Exponential)'] = {'Statistic': ks_stat_exp, 'P-value': ks_p_value_exp}

    # Anderson-Darling Test for Exponential Distribution
    ad_stat_exp, critical_values_exp, sig_level_exp = stats.anderson(data, dist='expon')
    results['Anderson-Darling Test (Exponential)'] = {'Statistic': ad_stat_exp, 'Critical Values': critical_values_exp}

    # Bootstrapping for Uniform Distribution
    bootstrapped_means = []
    for _ in range(1000):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(boot_sample))
    results['Bootstrapped Means (Uniform)'] = {'Mean': np.mean(bootstrapped_means)}

    return results

# Run tests
results = goodness_of_fit_tests(kmer_counts)

# Output results to a file
with open("goodness_of_fit.txt", 'w') as f:  # Use the standard file name
    # Write Kolmogorov-Smirnov test results
    f.write(f"KS Test (Uniform): Statistic={results['KS Test (Uniform)']['Statistic']}, P-value={results['KS Test (Uniform)']['P-value']}\n")
    f.write(f"KS Test (Exponential): Statistic={results['KS Test (Exponential)']['Statistic']}, P-value={results['KS Test (Exponential)']['P-value']}\n")

    # Write Anderson-Darling test results
    f.write(f"Anderson-Darling Test (Uniform): Statistic={results['Anderson-Darling Test (Uniform)']['Statistic']}, Critical Values={results['Anderson-Darling Test (Uniform)']['Critical Values']}\n")
    f.write(f"Anderson-Darling Test (Exponential): Statistic={results['Anderson-Darling Test (Exponential)']['Statistic']}, Critical Values={results['Anderson-Darling Test (Exponential)']['Critical Values']}\n")

    # Write Bootstrapping results
    f.write(f"Bootstrapped Means (Uniform): Mean={results['Bootstrapped Means (Uniform)']['Mean']}\n")

# Create a flag file to indicate the non-parametric test was completed
with open("nonparametric_test_done.flag", 'w') as flag_file:
    flag_file.write("Non-parametric goodness of fit test completed.\n")

print("Non-parametric goodness-of-fit test completed and flag created.")
