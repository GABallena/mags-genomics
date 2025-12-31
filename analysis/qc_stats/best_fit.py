#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

def analyze_fit_results(file_path, output_file):
    results = {}

    # Read the combined output file
    with open(file_path, 'r') as f:
        for line in f:
            if "P-value" in line:  # Only process lines with P-values
                parts = line.split(',')
                dist_name = parts[0].split(':')[0].strip()  # Get the distribution name
                p_value = float(parts[1].split('=')[1].strip())  # Extract the P-value
                results[dist_name] = p_value

    # Find the best fit based on the highest P-value
    if results:
        best_fit = max(results, key=results.get)  # Find the distribution with the highest P-value
        best_p_value = results[best_fit]

        # Write all tests and their P-values to the output file
        with open(output_file, 'w') as out_f:
            out_f.write("Distribution Test Results:\n")
            for dist_name, p_value in results.items():
                out_f.write(f"{dist_name}: P-value={p_value}\n")

            # Also write the best fit result
            out_f.write("\nBest fitting distribution:\n")
            out_f.write(f"{best_fit} with P-value={best_p_value}\n")

        print(f"Best fitting distribution: {best_fit} with P-value: {best_p_value}")
    else:
        print("No valid P-values found to determine the best fitting distribution.")

# Example usage
analyze_fit_results("goodness_of_fit.txt", "fit_results.tsv")
