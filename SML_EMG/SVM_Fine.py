#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:52:33 2023

@author: ariasarch
"""

# Import necessary packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

##############################################################################################################

import warnings
warnings.filterwarnings('ignore')

##############################################################################################################

# Create hyperparameter space
def get_hyperparameter_space():
    pbounds = {
        'C': (0.1, 100),
        'gamma': (0.1, 1)
    }
    return pbounds

# Run bayesian optimization 
def bayesian_optimization(evaluate_model, pbounds, n_iter):
    
    # Create optimizer
    optimizer = BayesianOptimization(f = evaluate_model, pbounds = pbounds, random_state = 42, verbose = 2, allow_duplicate_points=True)
    
    # Set GP parameters
    optimizer.set_gp_params(kernel = None, alpha = 1e-6)
    
    # Define utility function
    utility = UtilityFunction(kind = 'ucb', kappa = 2.5, xi = 0.0)
    
    # Print table header
    header = "Iter | Target | "
    for key in pbounds:
        header += f"{key} | "
    print(header)
    print("-" * len(header))

    # Maximize optimizer
    for i in range(n_iter):
        next_point_to_probe = optimizer.suggest(utility)
        target = evaluate_model(**next_point_to_probe)
        optimizer.register(params=next_point_to_probe, target=target)

        # Print table row with current iteration, target, and hyperparameters
        row = f"{i+1:4d} | {target:7.4f} | "
        for key in pbounds:
            row += f"{next_point_to_probe[key]:12.2f} | "
        print(row)

    print("Finished Bayesian optimization")
    
    # Train the final model with the best hyperparameters
    best_params = optimizer.max['params']
    print('\n', best_params)
    
    return best_params, optimizer, n_iter

# Run SVM model
def svm_model(best_params):
    model = SVC(C=best_params['C'], kernel='rbf', gamma=best_params['gamma'], probability=True)
    
    return model

# Plot each iteration's average 
def plot_avg(optimizer, n_iter, X_test, y_test, model, model_name):
    # Extract avg_score from each optimization run
    avg_scores = [run['target'] for run in optimizer.res]

    # Plot avg_score over iterations
    plt.plot(np.arange(1, n_iter+1), avg_scores)
    plt.xlabel('Iteration')
    plt.ylabel('Average Score')
    plt.ylim(min(avg_scores) - 0.0001, max(avg_scores) + 0.0001)  # Set y-axis limits
    plt.title(f'Bayesian Optimization Results for {model_name}')
    plt.show()

    # Make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy of Test Values: %.2f%%' % (accuracy * 100.0))

    return accuracy

##############################################################################################################

# Run Main Function
def exec_svm_fine(X_train, X_test, y_train, y_test):

    # Select only the numeric columns for scaling
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    # Scale the numeric columns using MinMaxScaler
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Reassign the column names to the scaled data
    X_train.columns = pd.Index(X_train.columns)
    X_test.columns = pd.Index(X_test.columns)

    # Call bounds for optimization 
    pbounds = get_hyperparameter_space()
    
    # Evaluate SVM model
    def evaluate_model(C, gamma):
        
        # Run SVM model
        model = SVC(C=C, kernel='rbf', gamma=gamma, probability=True)
    
        # Train and evaluate the model using cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring = 'accuracy')
        avg_score = cv_scores.mean()
        
        # Return the average accuracy
        return avg_score
    
    # Bayesian optimization
    best_params, optimizer, n_iter = bayesian_optimization(evaluate_model, pbounds, n_iter = 10)
    
    # Store the optimized model
    model = svm_model(best_params)
    
    # Run the SVM model 
    model.fit(X_train, y_train)
    
    # Plot accuracy 
    model_name = "SVM Fine"
    accuracy = plot_avg(optimizer, n_iter, X_test, y_test, model, model_name)
    
    model_type = "kernel"
    
    return model, accuracy, model_type, model_name

