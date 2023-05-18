#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:25:37 2023

@author: ariasarch
"""

# Import necessary packages 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

##############################################################################################################

# Run Main Function
def exec_nb_gaus(X_train, X_test, y_train, y_test):
    
    # Store the optimized model
    model = GaussianNB()
    
    # Run the model 
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Naïve Bayes - Gaussian")
    print("Accuracy: {:.2f}%".format(accuracy*100))
    
    model_type = "kernel"
    
    model_name = "Naïve Bayes Gaussian"
    
    return model, accuracy, model_type, model_name