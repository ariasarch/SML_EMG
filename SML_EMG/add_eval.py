#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:20:52 2023

@author: ariasarch
"""

import pandas as pd

def calculate_averages(dfs, base_path, filename, non_average_columns=['Model Name', 'Time']):
    # Concatenate all DataFrames
    all_models_df = pd.concat(dfs)

    # Group by the model name, and calculate the mean for each group
    avg_df = all_models_df.groupby('Model Name').mean()

    # Reset the index (because groupby creates a multi-index)
    avg_df.reset_index(inplace=True)

    # Create inverse for specified columns
    columns_to_inverse = ['Type I Error', 'Type II Error', 'False Omission Rate', 'False Discovery Rate']
    for col in columns_to_inverse:
        if col in avg_df.columns:
            avg_df[col] = 1 - avg_df[col]

    # Create a new 'Evaluation' column as the mean of all other columns (excluding the ones defined above)
    avg_df['Evaluation'] = avg_df.drop(columns=non_average_columns).mean(axis=1)

    # Sort the DataFrame by 'Evaluation' in descending order
    avg_df = avg_df.sort_values(by='Evaluation', ascending=False)

    # Reset the index (because sort_values creates a new index)
    avg_df.reset_index(drop=True, inplace=True)

    # Save the DataFrame with averages to a new CSV file
    avg_csv_filename = f'{base_path}{filename}'
    avg_df.to_csv(avg_csv_filename)
    
    return avg_df

def save_results_to_csv(dfs, shap_values_df, base_path):
    # Create a new DataFrame from the dfs list
    all_models_df = pd.concat(dfs)

    # Fill non-numeric columns with a suitable value
    all_models_df.loc['Mean_Metrics', ['Participant', 'Arm', 'Model', 'Model Name', 'Iteration', 'Model_Type']] = 'NaN'  # replace 'NaN' with the value you want

    # Calculate the mean of the metric columns
    mean_metrics = all_models_df.mean()

    # Append the mean metrics to the DataFrame
    all_models_df.loc['Mean_Metrics'] = mean_metrics

    # Save the DataFrame to disk as a single CSV file
    csv_filename = f'{base_path}Best_Models_Metrics.csv'
    all_models_df.to_csv(csv_filename)

    # Save SHAP values to a CSV file
    csv_filename = f'{base_path}Best_model_shap_values.csv'
    shap_values_df.to_csv(csv_filename)

def pick_best_model(avg_df):
    # Get the row with the highest evaluation
    best_row = avg_df.loc[avg_df['Evaluation'].idxmax()]

    # Extract the best model and its details
    best_model = best_row['Model Name']

    return best_model