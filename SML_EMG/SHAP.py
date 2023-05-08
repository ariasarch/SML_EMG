#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:49:28 2023

@author: ariasarch
"""
import shap
import numpy as np
import matplotlib.pyplot as plt

# SHAP values for Tree Models 
def shap_exp_tree(model, X_train, X_test):

    # Load JS visualization code to notebook
    shap.initjs()

    # Explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Calculate the average SHAP value for each feature, SHAP_i = (1/N) * Σ SHAP(x_j, i)
    SHAP = np.mean(np.abs(shap_values.values), axis=0)
    
    # Create a bar plot for sorted SHAP values
    fig, ax = plt.subplots(figsize=(8,6))

    # Get the indices of the sorted SHAP values in descending order
    idx = np.argsort(SHAP)

    # Plot the SHAP values in descending order
    ax.barh(X_train.columns[idx], SHAP[idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title('Feature Importance, Tree')
    plt.show()
    
    return SHAP

# Shap values for Kernel Models 
def shap_exp_kernel(model, X_train, X_test):

    # Load JS visualization code to notebook
    shap.initjs()

    # Set the number of background samples to use for SHAP value calculation
    background = shap.sample(X_train, 10)

    # Explain the model's predictions using SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test, nsamples=10)

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
    plt.show()

    return SHAP