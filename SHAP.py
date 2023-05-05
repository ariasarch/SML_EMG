#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:49:28 2023

@author: ariasarch
"""
import shap

# SHAP values for Tree Models 
def shap_exp_tree(model, X_train, X_test):
    
    # Load JS visualization code to notebook
    shap.initjs()

    # Explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model, X_train, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_test)

    # Summarize the effects of all the features
    shap.summary_plot(shap_values, X_test,  plot_type='bar', title = 'Individual Feature contribution to Model Prediction')

# Shap values for Kernel Models 
def shap_exp_kernel(model, X_train, X_test):
    
    # Load JS visualization code to notebook
    shap.initjs()
    
    # Set the number of background samples to use for SHAP value calculation
    background = shap.sample(X_train, 100)

    # Explain the model's predictions using SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test, nsamples=100)

    # Plot the SHAP values with feature names
    shap.summary_plot(shap_values, X_test, title = 'Individual Feature contribution to Model Prediction')