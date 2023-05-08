#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:14:30 2023

@author: ariasarch
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from SML_EMG.config import file_path

# Load the CSV file into a Pandas DataFrame 
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Split the data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load():
    # Call load_csv 
    df = load_csv(file_path)
    
    # Set the first column as y and the rest as X
    y = df['Stimulation']
    X = df.iloc[:, :-1]
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    return X_train, X_test, y_train, y_test