#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:07:07 2023

@author: ariasarch
"""

# Import all modules
import time
import pandas as pd
import pickle
import SML_EMG as se
from SML_EMG.config import base_path
from sklearn.metrics import confusion_matrix

# Run all SML models
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

# Run a single SML model
funcs = [se.exec_xgboost]

# List to store DataFrames for each model
model_search_df = []

# Loop over participants and arm side
for participant in range(11, 15):
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
        df_models = pd.DataFrame(columns=['Model Name', 'Model', 'Time', 'Iteration', 'Model_Type'])

        # Loop all SML functions
        for func in funcs:
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
               'Model': model, 
               'Model Name': model_name,
               'Time': elapsed_time, 
               'Iteration': iteration, 
               'Model_Type': model_type, 
               **metrics
            }, ignore_index=True)
            
            # Add iteration
            iteration += 1
            
            # Add the current DataFrame to the list
            model_search_df.append(df_models)
        
        # Save the DataFrame to disk as a CSV file
        csv_filename = f'{base_path}Participant_{participant}_{side}_models_sml.csv'
        df_models.to_csv(csv_filename)

# After finishing all models testing, evaluate them
average_df = se.calculate_averages(model_search_df, base_path, filename='All_Model_Averages.csv', non_average_columns=['Model Name', 'Time'])
best_model = se.pick_best_model(average_df)
print(best_model)