#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:50:43 2023

@author: ariasarch
"""

# Import necessary packages 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Run Main Function
def exec_lda(X_train, X_test, y_train, y_test):
    
    # Run the model
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Linear Discrminant")
    print("Accuracy:", accuracy)
    
    model_type = "kernel"
    
    model_name = "Linear Discrminant"
    
    return model, accuracy, model_type, model_name
