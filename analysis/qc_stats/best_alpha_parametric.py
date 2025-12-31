#!/usr/bin/env python3
# Portfolio-safe copy: distribution fitting / goodness-of-fit utilities
# Identifiers generalized; no private data included.

import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

# Check if the parametric goodness-of-fit test was completed
def check_parametric_test():
    if not os.path.isfile("parametric_test_done.flag"):
        print("Parametric goodness-of-fit test was not executed. Exiting...")
        return False
    return True

# 1. Youden's Index
def youdens_index(p_values, alpha_values, beta_values):
    youden_indices = []
    for alpha, beta in zip(alpha_values, beta_values):
        sensitivity = 1 - beta  # Power is 1 - beta
        specificity = 1 - alpha
        youden_index = sensitivity + specificity - 1
        youden_indices.append(youden_index)
    
    best_alpha_youden = alpha_values[np.argmax(youden_indices)]
    return best_alpha_youden, youden_indices

# 2. False Discovery Rate (Benjamini-Hochberg Procedure)
def fdr_bh(p_values, alpha):
    p_values_sorted = np.sort(p_values)
    n = len(p_values)
    fdr_thresholds = [(i / n) * alpha for i in range(1, n+1)]
    for i, (p_value, threshold) in enumerate(zip(p_values_sorted, fdr_thresholds)):
        if p_value > threshold:
            return fdr_thresholds[i-1] if i > 0 else alpha
    return alpha

# 3. Bayesian Decision Theory (Bayes Factor)
def bayes_factor(p_value):
    return (1 - p_value) / p_value  # Simplified Bayes Factor

# 4. ROC Curve Analysis
def roc_analysis(p_values, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, p_values)
    roc_auc = auc(fpr, tpr)
    best_threshold_index = np.argmax(tpr - fpr)
    best_alpha_roc = thresholds[best_threshold_index]
    return best_alpha_roc, roc_auc

# 5. Loss Function Minimization (Cost-Benefit Analysis)
def loss_function(alpha_values, type1_error_cost=1, type2_error_cost=1, beta_values=None):
    total_loss = []
    for alpha, beta in zip(alpha_values, beta_values):
        loss = type1_error_cost * alpha + type2_error_cost * beta
        total_loss.append(loss)
    
    best_alpha_loss = alpha_values[np.argmin(total_loss)]
    return best_alpha_loss, total_loss

# Function to load P-values from the goodness of fit test output
def load_p_values(file_path):
    p_values = []
    with open(file_path, 'r') as f:
        for line in f:
            # Assuming P-value is the second value in the line (e.g., "Normal: KS stat=..., P-value=...")
            p_value = float(line.split("P-value=")[1].strip())
            p_values.append(p_value)
    return p_values

# Main function to calculate optimal alpha thresholds for parametric data
def calculate_optimal_alpha_parametric(input_file):
    # Check if parametric test was executed
    if not check_parametric_test():
        return
    
    # Load P-values from goodness_of_fit.txt
    p_values = load_p_values(input_file)
    
    # Simulate alpha and beta values for testing purposes
    alpha_values = np.linspace(0.01, 0.1, 100)  # Range of possible alphas
    beta_values = np.linspace(0.01, 0.1, 100)  # Corresponding beta values for simplicity
    y_true = np.random.binomial(1, 0.5, 100)  # Simulated binary outcomes

    # 1. Youden's Index
    best_alpha_youden, youden_indices = youdens_index(p_values, alpha_values, beta_values)
    
    # 2. FDR (Benjamini-Hochberg Procedure)
    best_alpha_fdr = fdr_bh(p_values, alpha=0.05)
    
    # 3. Bayesian Decision Theory (Bayes Factors)
    bayes_factors = [bayes_factor(p) for p in p_values]
    
    # 4. ROC Curve Analysis
    best_alpha_roc, roc_auc = roc_analysis(p_values, y_true)
    
    # 5. Loss Function Minimization
    best_alpha_loss, total_loss = loss_function(alpha_values, beta_values=beta_values)

    # Presenting results
    best_alphas = {
        "Youden's Index": best_alpha_youden,
        "FDR (Benjamini-Hochberg)": best_alpha_fdr,
        "Bayesian Decision Theory": bayes_factors,
        "ROC Curve": best_alpha_roc,
        "Loss Function Minimization": best_alpha_loss
    }

    return best_alphas

# Example usage for parametric data
best_alphas_parametric = calculate_optimal_alpha_parametric("goodness_of_fit.txt")
if best_alphas_parametric:
    print("Best alphas for parametric data:", best_alphas_parametric)
