#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import pandas as pd
import json

# Load the fit results and extract best model and alpha
fit_results = pd.read_csv(snakemake.input[0], sep='\t')
best_fit = fit_results.loc[fit_results['P-value'].idxmax()]
best_model = best_fit['Model']
best_p_value = best_fit['P-value']

# Load alpha values and find the optimal alpha for the best-fit model
alpha_values = pd.read_csv(snakemake.input[1], sep='\t')
optimal_alpha = alpha_values.loc[alpha_values['Model'] == best_model, 'Optimal Alpha'].values[0]

# Save results to a JSON file for the next step
output = {
    "best_model": best_model,
    "optimal_alpha": optimal_alpha
}

with open(snakemake.output[0], 'w') as f:
    json.dump(output, f)
