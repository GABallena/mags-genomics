# Portfolio-safe copy: Snakemake workflow for distribution selection/validation.
# Identifiers generalized; no private data included.

# Final output, ensuring all previous rules run in sequence to produce this
rule all:
    input:
        "results/bestfit_and_alpha_output.json"  # The final output containing the best model and optimal alpha

# Step 1: Distribution Check (Normality Test)
rule check_distribution:
    input:
        "kmer_counts.txt"
    output:
        "distribution_check.txt"
    script:
        "scripts/normality_test.py"  # Run normality tests on k-mer counts to decide the distribution path

# Step 2: Fit Alternate Distributions
rule fit_alternate_distributions:
    input:
        "distribution_check.txt"
    output:
        "fitted_distributions.txt"  # Contains fitted distribution parameters (e.g., normal, log-normal, exponential)
    script:
        "scripts/fit_distributions.py"

# Step 3: Data Transformation (if needed based on fitted distributions)
rule transform_data_if_needed:
    input:
        "fitted_distributions.txt"
    output:
        "transformed_data.txt"  # Transformed data after fitting the distribution (e.g., log-transformation)
    script:
        "scripts/transform_data.py"

# Parametric Path - Goodness of Fit Tests
rule goodness_of_fit_parametric:
    input:
        "fitted_distributions.txt"
    output:
        "goodness_of_fit.txt",  # Contains the P-values and test statistics for the parametric distributions
        "parametric_test_done.flag"  # Flag to indicate completion of the parametric goodness-of-fit test
    script:
        "scripts/goodness_of_fit_parametric.py"

# Parametric Path - Best Alpha Determination
rule best_alpha_parametric:
    input:
        "goodness_of_fit.txt",
        "parametric_test_done.flag"  # Ensures the test was completed before this rule runs
    output:
        "best_alpha_parametric.txt"  # Contains the optimal alpha based on parametric results
    script:
        "scripts/best_alpha_parametric.py"

# Non-Parametric Path - Goodness of Fit Tests
rule goodness_of_fit_nonparametric:
    input:
        "kmer_counts.txt"
    output:
        "goodness_of_fit.txt",  # Contains the results of the non-parametric goodness-of-fit test
        "nonparametric_test_done.flag"  # Flag to indicate completion of the non-parametric goodness-of-fit test
    script:
        "scripts/goodness_of_fit_nonparametric.py"

# Non-Parametric Path - Best Alpha Determination
rule best_alpha_nonparametric:
    input:
        "goodness_of_fit.txt",
        "nonparametric_test_done.flag"  # Ensures the test was completed before this rule runs
    output:
        "best_alpha_nonparametric.txt"  # Contains the optimal alpha based on non-parametric results
    script:
        "scripts/best_alpha_nonparametric.py"

# Step 4: Best Fit Model Determination
rule best_fit:
    input:
        "goodness_of_fit.txt"  # Results of goodness-of-fit tests
    output:
        "fit_results.tsv"  # Contains the best fitting model and its associated P-value
    script:
        "scripts/best_fit.py"

# Final Step: Combine Best Fit and Alpha Values
rule bestfit_and_alpha:
    input:
        fit_results="results/fit_results.tsv",  # Best model and P-value
        alpha_values="results/alpha_values.tsv"  # Optimal alpha values for each model
    output:
        "results/bestfit_and_alpha_output.json"  # JSON file containing the best model and optimal alpha
    script:
        "scripts/bestfit_and_alpha.py"  # Script that determines the final best-fit model and alpha
