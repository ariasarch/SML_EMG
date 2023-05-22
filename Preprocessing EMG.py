#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  19 09:47:30 2023

@author: ariasarch
"""

# Import necessary packages
import os
import pandas as pd
import matplotlib.pyplot as plt

# Create function to read and reorder csv file
def txt_file(file_path):

    # Read the csv file into a dataframe
    df = pd.read_csv(file_path, delimiter='\t')

    df = df.drop(['Channel 3'], axis=1)

    # Add a new column 'time' where its value is equal to the number of rows 
    df['time'] = range(1, len(df) + 1)

    return df

# Plot time vs prediction signal
def plot_pred(df):
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['Stimulation'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stimulation')
    ax.set_title('Stimulation vs Time')
    plt.show()

# Get column stats 
def get_column_stats(df, col_name):
    col_min = df[col_name].min()
    col_max = df[col_name].max()
    col_mean = df[col_name].mean()

    return (col_min, col_max, col_mean)

# Apply moving average smoothing to a column
def smooth_column(df, column_name, window_size):
    # Create a new column with the smoothed values
    df[column_name] = df[column_name].rolling(window_size, center=True, min_periods=1).mean()
    return df

# Apply smoothing
def threshold(x, mean):

    # Set the threshold
    if x < mean:
        return 0
    else:
        return 1
    
# Define the base path for input and output
input_base_path = '/Users/ariasarch/Desktop/'
output_base_path = '/Users/ariasarch/Desktop/'

# Loop over participants and arm side
for participant in range(11, 16):
    for side in ['Right', 'Left']:
        # Prepare the file paths for input and output
        input_file_path = f'{input_base_path}Participant_{participant}_SML_EMG_{side}.txt'
        output_file_path = f'{output_base_path}Participant_{participant}_SML_EMG_{side}_Processed.csv'

        if os.path.exists(input_file_path):  # check if file exists
            # Call txt file
            df = txt_file(input_file_path)
            
            # Apply moving average smoothing to the 'Stimulation' column
            df = smooth_column(df, 'Stimulation', 500)

            # Get stats
            col_min, col_max, col_mean = get_column_stats(df, 'Stimulation')

            # Apply smoothing
            df['Stimulation'] = df['Stimulation'].apply(threshold, mean=df['Stimulation'].mean())

            # Drop the time column
            df = df.drop('time', axis=1)

            # Save modified DataFrame as a new CSV file
            df.to_csv(output_file_path, index=False)
