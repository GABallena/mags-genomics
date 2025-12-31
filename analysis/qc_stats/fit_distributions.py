#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, gamma

# Load k-mer counts
kmer_counts = np.loadtxt(snakemake.input[0])

# Fit distributions
norm_params = norm.fit(kmer_counts)
lognorm_params = lognorm.fit(kmer_counts)
expon_params = expon.fit(kmer_counts)
gamma_params = gamma.fit(kmer_counts)

# Plot the histogram
plt.hist(kmer_counts, bins=50, density=True, alpha=0.6, color='blue')

# Plot the fitted distributions
x = np.linspace(min(kmer_counts), max(kmer_counts), 100)

plt.plot(x, norm.pdf(x, *norm_params), label="Normal", color='red')
plt.plot(x, lognorm.pdf(x, *lognorm_params), label="Log-normal", color='green')
plt.plot(x, expon.pdf(x, *expon_params), label="Exponential", color='purple')
plt.plot(x, gamma.pdf(x, *gamma_params), label="Gamma", color='orange')

plt.legend()
plt.savefig(snakemake.output[0])
plt.close()

# Write fitted parameters to file
with open(snakemake.output[1], 'w') as f:
    f.write(f"Normal params: {norm_params}\n")
    f.write(f"Log-normal params: {lognorm_params}\n")
    f.write(f"Exponential params: {expon_params}\n")
    f.write(f"Gamma params: {gamma_params}\n")
