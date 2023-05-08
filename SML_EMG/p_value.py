#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:03:59 2023

@author: ariasarch
"""

# Import the necessary packages 
import shap
import numpy as np
import pandas as pd

# Set global variables
n_permutations = 10
p_values = []

def p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP):

    # Perform a permutation test 
    for _ in range(n_permutations):
        
        # Shuffle the y values
        y_perm = np.random.permutation(y_train)
    
        # Train the model with permuted y values
        model_perm = best_model.fit(X_train, y_perm)
    
        # Compute the SHAP values 
        explainer_perm = shap.TreeExplainer(model_perm, X_train)
        SHAP_perm_values = explainer_perm(X_test, check_additivity=False)
       
        # Calculate the average SHAP value for the permutation, SHAP_perm_i,b = (1/N) * Σ SHAP_perm(x_j, i)
        SHAP_perm = np.mean(np.abs(SHAP_perm_values.values), axis=0)
    
        # Calculate the P value component for the current permutation, ΔSHAP_i,b = SHAP_perm_i,b - SHAP_i, P_i = (1/B) * Σ I(|ΔSHAP_i,b| >= |SHAP_i - μ(SHAP_i)|)
        p_values.append(abs(SHAP_perm - SHAP) >= abs(SHAP - 0))
    
    # Calculate the final P value for each feature
    P_values = np.mean(p_values, axis=0)
    
    # Calculate the standard error of the P value estimate for each feature
    SEs = np.sqrt(P_values * (1 - P_values) / n_permutations)
    
    # Compare the P values to the pre-determined significance level (e.g., α = 0.05)
    alpha = 0.05
    
    # Apply Bonferroni correction to the significance level
    corrected_alpha = alpha / len(P_values)
    
    significant_features = P_values < corrected_alpha
    
    # Retrieve feature names
    feature_names = X_test.columns.tolist() 
    
    # Create a df of p values, se, and features 
    p_vals = {
        "Feature Name": feature_names,
        "Shap Values": SHAP,
        "P-value": P_values,
        "Standard Error": SEs,
        "Significance": ["Significant" if s else "Not significant" for s in significant_features]
    }
    
    # Convert to pd df
    results_df = pd.DataFrame(p_vals)
    
    # Apply scientific notation format to the "P-value" column
    results_df['P-value'] = results_df['P-value'].apply('{:.1e}'.format)

    # Sort the DataFrame by the highest SHAP value
    sorted_results_df = results_df.sort_values(by="Shap Values", ascending=False)
    
    # Display the sorted DataFrame
    print(sorted_results_df)
    
def p_value_kernel(X_train, X_test, y_train, y_test, best_model, SHAP):

    # Perform a permutation test 
    for _ in range(n_permutations):
        
        # Shuffle the y values
        y_perm = np.random.permutation(y_train)
    
        # Train the model with permuted y values
        model_perm = best_model.fit(X_train, y_perm)
    
        # Set the number of background samples to use for SHAP value calculation
        background = shap.sample(X_train, 10)

        # Explain the model's predictions using SHAP values
        explainer_perm = shap.KernelExplainer(model_perm.predict_proba, background)
        SHAP_perm_values = explainer_perm.shap_values(X_test, nsamples=10)
        
        #Convert the list of arrays to a single array
        SHAP_perm_values = np.vstack(SHAP_perm_values)
       
        # Calculate the average SHAP value for the permutation, SHAP_perm_i,b = (1/N) * Σ SHAP_perm(x_j, i)
        SHAP_perm = np.mean(np.abs(SHAP_perm_values.values), axis=0)
    
        # Calculate the P value component for the current permutation, ΔSHAP_i,b = SHAP_perm_i,b - SHAP_i, P_i = (1/B) * Σ I(|ΔSHAP_i,b| >= |SHAP_i - μ(SHAP_i)|)
        p_values.append(abs(SHAP_perm - SHAP) >= abs(SHAP - 0))
    
    # Calculate the final P value for each feature
    P_values = np.mean(p_values, axis=0)
    
    # Calculate the standard error of the P value estimate for each feature
    SEs = np.sqrt(P_values * (1 - P_values) / n_permutations)
    
    # Compare the P values to the pre-determined significance level (e.g., α = 0.05)
    alpha = 0.05
    
    # Apply Bonferroni correction to the significance level
    corrected_alpha = alpha / len(P_values)
    
    significant_features = P_values < corrected_alpha
    
    # Retrieve feature names
    feature_names = X_test.columns.tolist() 
    
    # Create a df of p values, se, and features 
    p_vals = {
        "Feature Name": feature_names,
        "Shap Values": SHAP,
        "P-value": P_values,
        "Standard Error": SEs,
        "Significance": ["Significant" if s else "Not significant" for s in significant_features]
    }
    
    # Convert to pd df
    results_df = pd.DataFrame(p_vals)
    
    # Apply scientific notation format to the "P-value" column
    results_df['P-value'] = results_df['P-value'].apply('{:.1e}'.format)

    # Sort the DataFrame by the highest SHAP value
    sorted_results_df = results_df.sort_values(by="Shap Values", ascending=False)
    
    # Display the sorted DataFrame
    print(sorted_results_df)