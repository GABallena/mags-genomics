#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

from scipy.stats import kstest, norm, lognorm, expon, gamma

# Load k-mer counts
kmer_counts = np.loadtxt(snakemake.input[0])

# Fit parameters from the fit_distributions step
with open(snakemake.input[1]) as f:
    params = [eval(line.split(": ")[1]) for line in f.readlines()]

norm_params, lognorm_params, expon_params, gamma_params = params

# Perform KS test for each distribution and handle any potential errors
try:
    ks_stat_norm, p_value_norm = kstest(kmer_counts, 'norm', args=norm_params)
except Exception as e:
    ks_stat_norm, p_value_norm = None, None
    print(f"Error in KS test for Normal distribution: {e}")

try:
    ks_stat_lognorm, p_value_lognorm = kstest(kmer_counts, 'lognorm', args=lognorm_params)
except Exception as e:
    ks_stat_lognorm, p_value_lognorm = None, None
    print(f"Error in KS test for Log-normal distribution: {e}")

try:
    ks_stat_expon, p_value_expon = kstest(kmer_counts, 'expon', args=expon_params)
except Exception as e:
    ks_stat_expon, p_value_expon = None, None
    print(f"Error in KS test for Exponential distribution: {e}")

try:
    ks_stat_gamma, p_value_gamma = kstest(kmer_counts, 'gamma', args=gamma_params)
except Exception as e:
    ks_stat_gamma, p_value_gamma = None, None
    print(f"Error in KS test for Gamma distribution: {e}")

# Write results to a file
with open("goodness_of_fit.txt", 'w') as f:
    if ks_stat_norm is not None:
        f.write(f"Normal: KS stat={ks_stat_norm}, P-value={p_value_norm}\n")
    else:
        f.write("Normal: Error in KS test\n")
    
    if ks_stat_lognorm is not None:
        f.write(f"Log-normal: KS stat={ks_stat_lognorm}, P-value={p_value_lognorm}\n")
    else:
        f.write("Log-normal: Error in KS test\n")
    
    if ks_stat_expon is not None:
        f.write(f"Exponential: KS stat={ks_stat_expon}, P-value={p_value_expon}\n")
    else:
        f.write("Exponential: Error in KS test\n")
    
    if ks_stat_gamma is not None:
        f.write(f"Gamma: KS stat={ks_stat_gamma}, P-value={p_value_gamma}\n")
    else:
        f.write("Gamma: Error in KS test\n")

# Write results to goodness_of_fit.txt (same as before)
with open("goodness_of_fit.txt", 'w') as f:
    f.write(f"Normal: KS stat={ks_stat_norm}, P-value={p_value_norm}\n")
    f.write(f"Log-normal: KS stat={ks_stat_lognorm}, P-value={p_value_lognorm}\n")
    f.write(f"Exponential: KS stat={ks_stat_expon}, P-value={p_value_expon}\n")
    f.write(f"Gamma: KS stat={ks_stat_gamma}, P-value={p_value_gamma}\n")

# Create a flag file to indicate the parametric test was completed
with open("parametric_test_done.flag", 'w') as flag_file:
    flag_file.write("Parametric goodness of fit test completed.\n")

print("Parametric goodness-of-fit test completed and flag created.")
