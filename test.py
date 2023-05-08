#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:07:07 2023

@author: ariasarch
"""

# Import all modules
import time
import SML_EMG as se

# Split and load the data
X_train, X_test, y_train, y_test = se.load()

# Run all SML models
# funcs = [se.exec_xgboost, se.exec_logitboost, se.exec_adaboost, 
#           se.exec_decision_trees,
#           se.exec_lda, se.exec_qda,
#           se.exec_nb_gaus, se.exec_nb_bern, se.exec_nb_multi,
#           se.exec_svm_linear, se.exec_svm_quadratic, se.exec_svm_cubic, se.exec_svm_fine, se.exec_svm_medium, se.exec_svm_coarse,
#           se.exec_KNN_fine, se.exec_KNN_medium, se.exec_KNN_coarse, se.exec_KNN_cubic, se.exec_KNN_weighted, se.exec_KNN_cosine,
#           se.exec_random_forest, se.exec_extra_trees]

# Run a single SML model
funcs = [se.exec_xgboost]

# Store all models and metrics 
dic = {}
iteration = 0

# Loop all SML functions
for func in funcs:
    print("Running:", func.__name__, "for iteration", iteration)
    
    # Start time
    start_time = time.time()
    
    # Run model
    model, accuracy, model_type = func(X_train, X_test, y_train, y_test)
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Add to dictionary 
    dic[accuracy] = (model, elapsed_time, iteration, model_type)
    
    # Add iteration
    iteration += 1

# Call 'best' SML 
key = sorted(list(dic.keys()), reverse=True)[0]
best_model, best_time, best_iter, best_model_type = dic[key]
print(f"Best model: {best_model}, \n with accuracy: {key}, and time taken: {best_time:.2f} seconds and iteration: {best_iter}")

# Calculate SHAP Values
if model_type == "tree":
    SHAP = se.shap_exp_tree(best_model, X_train, X_test)
    
    # uncomment for p values
    
    # se.p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP)
else:
    SHAP = se.shap_exp_kernel(best_model, X_train, X_test)
    
    # uncomment for p values
    
    # se.p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP)