#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:35:37 2023

@author: ariasarch
"""

# Import necessary packages
import time
import pandas as pd
from SML_EMG.config import base_path
import SML_EMG as se
from sklearn.metrics import confusion_matrix

# List of all SML models used
# funcs = [se.exec_xgboost, se.exec_logitboost, se.exec_adaboost, 
#           # se.exec_lightgbm,
#           se.exec_decision_trees,
#           se.exec_lda, se.exec_qda,
#           se.exec_nb_gaus, se.exec_nb_bern, se.exec_nb_multi,
#           se.exec_svm_linear,
#           se.exec_svm_fine, se.exec_svm_medium, se.exec_svm_coarse,
#           # se.exec_svm_quadratic, se.exec_svm_cubic, 
#           se.exec_KNN_fine, se.exec_KNN_medium, se.exec_KNN_coarse, se.exec_KNN_cubic, se.exec_KNN_weighted, se.exec_KNN_quadratic,
#           se.exec_random_forest, se.exec_extra_trees,
#           se.exec_FNN]

# Pick the best model
func = se.exec_extra_trees

# List to store DataFrames for each model
dfs = []

# Initialize empty DataFrame for SHAP values
shap_values_df = pd.DataFrame()

# Loop over participants and arm side
for participant in range(1, 16):
    for side in ['Right', 'Left']:
        
        # Keep track of loop
        print(f"Processing Participant {participant}, Side: {side}")        

        # Prepare the file path
        file_path = f'{base_path}Participant_{participant}_SML_EMG_{side}_Processed.csv'
        
        # Split and load the data
        X_train, X_test, y_train, y_test = se.load(file_path)
            
        # Store iteration
        iteration = 0
        
        # Initialize an empty DataFrame to store results for this participant and side
        df_models = pd.DataFrame(columns=['Participant', 'Arm', 'Model Name', 'Model', 'Time', 'Iteration', 'Model_Type'])
        df_shap = pd.DataFrame(columns=['Participant', 'Arm'])
        
        # Run best SML function
        print("Running:", func.__name__, "for iteration", iteration)
        
        # Start time
        start_time = time.time()
        
        # Run model
        model, accuracy, model_type, model_name = func(X_train, X_test, y_train, y_test)
        
        # End time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get the predicted labels from the model
        y_pred = model.predict(X_test).flatten()
        y_pred = [round(value) for value in y_pred]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate the metrics
        metrics = se.calculate_metrics(tp, fn, fp, tn)
        
        # Append to DataFrame 
        df_models = df_models.append({
           'Participant': participant,
           'Arm': side,
           'Model': model, 
           'Model Name': model_name,
           'Time': elapsed_time, 
           'Iteration': iteration, 
           'Model_Type': model_type, 
           **metrics
        }, ignore_index=True)
       
        # Add the current DataFrame to the list
        dfs.append(df_models)
        
        # Compute and save SHAP values
        if model_type == "tree":
            
            SHAP = se.shap_exp_tree(model, X_train, X_test)
            
            # uncomment for p values
                
            # se.p_value_tree(X_train, X_test, y_train, y_test, best_model, SHAP)
            
        elif model_type == "kernel":
            
            SHAP = se.shap_exp_kernel(model, X_train, X_test)
            
            # uncomment for p values
                
            # se.p_value_kernel(X_train, X_test, y_train, y_test, best_model, SHAP)
        
        # Convert SHAP values to a DataFrame, append participant and side information
        df_shap = pd.DataFrame(SHAP, columns=X_train.columns)
        df_shap['Participant'] = participant
        df_shap['Arm'] = side
        
        # Append the new DataFrame to the main DataFrame
        shap_values_df = shap_values_df.append(df_shap, ignore_index=True)
        
        # Add iteration
        iteration += 1

se.save_results_to_csv(dfs, shap_values_df, base_path)
