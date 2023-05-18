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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Split and load the data
X_train, X_test, y_train, y_test = se.load()

# Run all SML models
# funcs = [se.exec_xgboost, se.exec_logitboost, se.exec_adaboost, 
#           # se.exec_lightgbm,
#           se.exec_decision_trees,
#           se.exec_lda, se.exec_qda,
#           se.exec_nb_gaus, se.exec_nb_bern, se.exec_nb_multi,
#           se.exec_svm_linear,se.exec_svm_fine, se.exec_svm_medium, se.exec_svm_coarse,
#           # se.exec_svm_quadratic, se.exec_svm_cubic, 
#           se.exec_KNN_fine, se.exec_KNN_medium, se.exec_KNN_coarse, se.exec_KNN_cubic, se.exec_KNN_weighted, se.exec_KNN_cosine,
#           se.exec_random_forest, se.exec_extra_trees,
#           se.exec_FNN]

# Run a single SML model
funcs = [se.exec_xgboost]

# Store iteration
iteration = 0

# Initialize an empty DataFrame to store results
df_models = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'Time', 'Iteration', 'Model_Type'])

# Loop all SML functions
for func in funcs:
    print("Running:", func.__name__, "for iteration", iteration)
    
    # Start time
    start_time = time.time()
    
    # Run model
    model, accuracy, model_type, model_name = func(X_train, X_test, y_train, y_test)
    
    # Get the predicted labels from the model
    y_pred = model.predict(X_test).flatten()
    y_pred = [round(value) for value in y_pred]
    
    # Calculate precision
    precision = precision_score(y_test, y_pred)
    
    # Calculate recall
    recall = recall_score(y_test, y_pred)
    
    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_test, y_pred)
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Append to DataFrame 
    df_models = df_models.append({
       'Model': model, 
       'Model Name': model_name,
       'Accuracy': accuracy,
       'Precision': precision,
       'Recall': recall,
       'F1-score': f1,
       'AUC-ROC': auc_roc,
       'Time': elapsed_time, 
       'Iteration': iteration, 
       'Model_Type': model_type
    }, ignore_index=True)
    
    # Save the model to disk
    # model_filename = func.__name__.replace("exec_", "") + '_model.sav'
    # pickle.dump(model, open(model_filename, 'wb'))
    
    # Add iteration
    iteration += 1

# Save the DataFrame to disk as a CSV file
csv_filename = 'models_sml.csv'
df_models.to_csv(csv_filename)
    