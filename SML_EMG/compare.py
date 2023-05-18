#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:43:35 2023

@author: ariasarch
"""

# Get the row with the highest accuracy
best_row = df_models.loc[df_models['Accuracy'].idxmax()]

# Extract the best model and its details
best_model = best_row['Model']
best_time = best_row['Time']
best_iter = best_row['Iteration']
best_model_type = best_row['Model_Type']
best_accuracy = best_row['Accuracy']

print(f"Best model: {best_model}, \n with accuracy: {best_accuracy}, and time taken: {best_time:.2f} seconds and iteration: {best_iter}")

# Calculate SHAP Values
if model_type == "tree":
    SHAP = se.shap_exp_tree(best_model, X_train, X_test)
    
    # uncomment for p values
    
    # se.p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP)
else:
    SHAP = se.shap_exp_kernel(best_model, X_train, X_test)
    
    # uncomment for p values
    
    # se.p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP)
    
# Save the SHAP values to disk
shap_filename = 'shap_values_sml.sav'
# pickle.dump(SHAP, open(shap_filename, 'wb'))  
    
# Load models if need be
#loaded_models, loaded_dic, loaded_SHAP = load_models_and_data(funcs)