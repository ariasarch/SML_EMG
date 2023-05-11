#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:44:20 2023

@author: ariasarch
"""

# Import necessary packages 
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

##############################################################################################################

import warnings
warnings.filterwarnings('ignore')

##############################################################################################################

# Create hyperparameter space
def get_hyperparameter_space():
    pbounds = {
        'max_depth': (5, 15),
        'subsample': (0.1, 1.0),
        'n_estimators': (1, 100),
        'learning_rate': (0.01, 0.99)
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

# Run xgboost model
def xgboost_model(best_params):
    model = XGBClassifier(max_depth=int(best_params['max_depth']), subsample=best_params['subsample'],
                          n_estimators=int(best_params['n_estimators']), learning_rate=best_params['learning_rate'],
                          objective='binary:logistic', eval_metric='error'
                          )

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
def exec_xgboost(X_train, X_test, y_train, y_test):

    # Call bounds for optimization 
    pbounds = get_hyperparameter_space()
    
    # Evaluate XGBoost model
    def evaluate_model(max_depth, subsample, n_estimators, learning_rate):
        
        # Define the model with the given hyperparameters
        model = XGBClassifier(
            max_depth = int(max_depth),
            subsample = subsample,
            n_estimators = int(n_estimators),
            learning_rate = learning_rate,
            objective = 'binary:logistic',
            eval_metric = 'error'
        )
        
        # Train and evaluate the model using cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring = 'accuracy')
        avg_score = cv_scores.mean()
        
        # Return the average accuracy
        return avg_score
    
    # Bayesian optimization
    best_params, optimizer, n_iter = bayesian_optimization(evaluate_model, pbounds, n_iter = 10)
    
    # Store the optimized model
    model = xgboost_model(best_params)
    
    # Run the xgboost model 
    model.fit(X_train, y_train)
    
    # Plot accuracy 
    model_name = "XGBoost"
    accuracy = plot_avg(optimizer, n_iter, X_test, y_test, model, model_name)
    
    model_type = "tree"
    
    return model, accuracy, model_type