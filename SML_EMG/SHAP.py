#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:49:28 2023

@author: ariasarch
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shap_exp_tree(model, X_train, X_test, participant, side):
    # Load JS visualization code to notebook
    shap.initjs()

    # Explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, X_train)   
    shap_values = explainer(X_test, check_additivity=False)
    
    # Calculate the average SHAP value for each feature, SHAP_i = (1/N) * Σ SHAP(x_j, i)
    SHAP = np.mean(np.abs(shap_values.values), axis=0)
    
    # For binary classification ensure only the first column is kept 
    if len(SHAP.shape) == 2:
        SHAP = SHAP[:, 0]
       
    # Create a bar plot for sorted SHAP values
    fig, ax = plt.subplots(figsize=(8,6))

    # Get the indices of the sorted SHAP values in descending order
    idx = np.argsort(SHAP)

    # Plot the SHAP values in descending order
    ax.barh(X_train.columns[idx], SHAP[idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title('Feature Importance, Tree')
    
    # Adjust the margins
    plt.tight_layout()
    
    # Construct the file path
    file_path = f"/Users/ariasarch/Desktop/shap_plot_participant_{participant}_{side}_tree.png"
    
    # Save the figure
    plt.savefig(file_path)
    
    plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)
    
    # Ensure SHAP is a 2D array
    SHAP = SHAP.reshape(1, -1)
    
    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(SHAP, columns=X_train.columns, index=['SHAP'])
    
    # Save the DataFrame to disk as a CSV file
    csv_file_path = f"/Users/ariasarch/Desktop/shap_values_tree_{participant}_{side}.csv"
    shap_df.to_csv(csv_file_path)
    
    return SHAP

# SHAP values for Kernel Models 
def shap_exp_kernel(model, X_train, X_test, participant, side):
    # Load JS visualization code to notebook
    shap.initjs()

    # Set the number of background samples to use for SHAP value calculation
    background = shap.sample(X_train, 100)

    # Explain the model's predictions using SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test, nsamples=100, check_additivity=False)

    # Convert the list of arrays to a single array
    shap_values = np.vstack(shap_values)

    # Calculate the average SHAP value for each feature, SHAP_i = (1/N) * Σ SHAP(x_j, i)
    SHAP = np.mean(np.abs(shap_values), axis=0)

    # Create a bar plot for sorted SHAP values
    fig, ax = plt.subplots(figsize=(8,6))

    # Get the indices of the sorted SHAP values in descending order
    idx = np.argsort(SHAP)

    # Plot the SHAP values in descending order
    ax.barh(range(X_train.shape[1]), SHAP[idx])
    ax.set_yticks(range(X_train.shape[1]))
    ax.set_yticklabels(X_train.columns[idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title('Feature Importance, Kernel')
    
    # Adjust the margins
    plt.tight_layout()
    
    # Construct the file path
    file_path = f"/Users/ariasarch/Desktop/shap_plot_participant_{participant}_{side}_tree.png"
    
    # Save the figure
    plt.savefig(file_path)
    
    plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)
    
    # Ensure SHAP is a 2D array
    SHAP = SHAP.reshape(1, -1)
    
    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(SHAP, columns=X_train.columns, index=['SHAP'])
    
    # Save the DataFrame to disk as a CSV file
    shap_df.to_csv('shap_values_kernel_sml.csv')

    return SHAP