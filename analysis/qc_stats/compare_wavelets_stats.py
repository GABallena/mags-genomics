#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, pearsonr, ks_2samp, f_oneway
import json

# Load wavelet-normalized data
contaminants = np.loadtxt(snakemake.input.contam)
scgs = np.loadtxt(snakemake.input.scgs)

# Load best-fit and alpha results
with open(snakemake.input.bestfit_alpha, 'r') as f:
    bestfit_alpha = json.load(f)

best_model = bestfit_alpha['best_model']
optimal_alpha = bestfit_alpha['optimal_alpha']

# Store results for multiple tests
test_results = {}

# Choose statistical tests based on the best model
if best_model == 'normal':
    # Parametric Tests
    # 1. Perform t-test (Independent)
    t_stat, t_p_value = stats.ttest_ind(contaminants, scgs)
    test_results['t-test'] = (t_stat, t_p_value)

    # 2. Perform One-way ANOVA (if there are more than 2 groups)
    if len(contaminants.shape) > 1:
        f_stat, anova_p_value = stats.f_oneway(contaminants, scgs)
        test_results['one-way ANOVA'] = (f_stat, anova_p_value)

    # 3. Pearson Correlation
    corr_coeff, corr_p_value = pearsonr(contaminants, scgs)
    test_results['Pearson correlation'] = (corr_coeff, corr_p_value)

elif best_model in ['lognormal', 'exponential', 'gamma', 'other_non_normal']:
    # Non-Parametric Tests
    # 1. Perform Mann-Whitney U test
    u_stat, mw_p_value = mannwhitneyu(contaminants, scgs, alternative='two-sided')
    test_results['Mann-Whitney U'] = (u_stat, mw_p_value)

    # 2. Perform Wilcoxon Signed-Rank test (if paired data is expected)
    if contaminants.shape == scgs.shape:
        w_stat, wilcox_p_value = wilcoxon(contaminants, scgs)
        test_results['Wilcoxon Signed-Rank'] = (w_stat, wilcox_p_value)

    # 3. Spearman Correlation
    spearman_coeff, spearman_p_value = spearmanr(contaminants, scgs)
    test_results['Spearman correlation'] = (spearman_coeff, spearman_p_value)

# 4. Kolmogorov-Smirnov Test (for comparing distributions)
ks_stat, ks_p_value = ks_2samp(contaminants, scgs)
test_results['KS Test'] = (ks_stat, ks_p_value)

# Bonferroni and FDR corrections
p_values = np.array([result[1] for result in test_results.values()])

# Bonferroni correction
bonferroni_corrected_p_values = np.minimum(p_values * len(p_values), 1.0)

# FDR correction (Benjamini-Hochberg)
sorted_p_indices = np.argsort(p_values)
sorted_p_values = p_values[sorted_p_indices]
n_tests = len(sorted_p_values)
fdr_thresholds = [(i / n_tests) * optimal_alpha for i in range(1, n_tests + 1)]
fdr_corrected_p_values = np.minimum.accumulate(sorted_p_values / fdr_thresholds)

# Map corrected p-values back to the original test results
for idx, test in enumerate(test_results.keys()):
    original_stat, original_p_value = test_results[test]
    bonferroni_p = bonferroni_corrected_p_values[idx]
    fdr_p = fdr_corrected_p_values[idx]
    test_results[test] = (original_stat, original_p_value, bonferroni_p, fdr_p)

# Write the results to the output files
with open(snakemake.output[0], 'w') as f:
    for test, (stat, p_value, bonferroni_p, fdr_p) in test_results.items():
        f.write(f"{test}:\n")
        f.write(f"Test statistic: {stat}\n")
        f.write(f"Original P-value: {p_value}\n")
        f.write(f"Bonferroni-corrected P-value: {bonferroni_p}\n")
        f.write(f"FDR-corrected P-value: {fdr_p}\n")
        f.write("\n")

# Save ambiguous sequences if P-values are above the threshold
with open(snakemake.output.ambiguous_sequences, 'w') as f:
    for idx, (test, (_, p_value, bonferroni_p, fdr_p)) in enumerate(test_results.items()):
        if p_value < optimal_alpha or bonferroni_p < optimal_alpha or fdr_p < optimal_alpha:
            f.write(f"Sequence {idx} is ambiguous for test {test}\n")
