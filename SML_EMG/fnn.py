#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:40:52 2023

@author: ariasarch
"""

# Import necessary packages 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.activations import sigmoid, relu, tanh
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow.keras.optimizers
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

##############################################################################################################

# Create hyperparameter space
def get_hyperparameter_space():
    pbounds = {
        'batch_size': (10, 500),
        'epochs': (10, 100),
        'learning_rate': (0.001, 0.1),
        'dropout_rate': (0.0, 0.5),
        'l1_reg': (0.0, 0.1),
        'l2_reg': (0.0, 0.1)
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
    y_pred = model.predict(X_test).flatten()
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy of Test Values: %.2f%%' % (accuracy * 100.0))

    return accuracy

##############################################################################################################

# Run Main Function
def exec_FNN(X_train, X_test, y_train, y_test):

    # Select only the numeric columns for scaling
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    # Scale the numeric columns using MinMaxScaler
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Reassign the column names to the scaled data
    X_train.columns = pd.Index(X_train.columns)
    X_test.columns = pd.Index(X_test.columns)

    # Define the list of activation functions and optimizers
    activation_functions = ['sigmoid', 'relu', 'tanh']
    optms = ['SGD', 'Adam', 'RMSprop', 'Adagrad']
    
    best_accuracy = 0.0
    best_activation = None
    best_optimizer = None
    
    # Print table header
    header = "Iter | Accuracy | Activation Func | Optimizer"
    print(header)
    print("-" * len(header))
    
    iteration_counter = 0
    
    # Iterate over all combinations of optimizers and activation functions
    for opt in optms:
        
        for activation in activation_functions:
    
            optm = tensorflow.keras.optimizers.get(opt) # creating a new instance of the optimizer
    
            # Create a Sequential Neural Network
            nn = Sequential()
            nn.add(Dense(32, input_dim=X_train.shape[1], activation=activation, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            nn.add(Dropout(0.5))  # 50% dropout
            nn.add(Dense(16, activation=activation, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            nn.add(Dropout(0.5))  # 50% dropout
            nn.add(Dense(1, activation='sigmoid'))
    
            # Compile the model with the optimizer
            nn.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])
    
            # Train the NN
            nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
            # Evaluate the NN model
            _, accuracy = nn.evaluate(X_test, y_test, verbose=0)
    
            # Print current iteration, accuracy, activation function, and optimizer
            print(f"{iteration_counter + 1:4d} | {accuracy:8.4f} | {activation:16s} | {opt}")
    
            # Check if the current combination is the best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_activation = activation
                best_optimizer = opt
    
            iteration_counter += 1
    
    print(best_accuracy)
    print(best_activation)
    print(best_optimizer)
      
    ##############################################################################################################

    # Run FNN model
    def fnn_model(best_params):
            
            # Create a Sequential Neural Network
            fnn = Sequential()
            fnn.add(Dense(32, input_dim=X_train.shape[1], activation=best_activation, kernel_regularizer=l1_l2(l1=best_params['l1_reg'], l2=best_params['l2_reg'])))
            fnn.add(Dropout(best_params['dropout_rate'])) 
            fnn.add(Dense(16, activation=best_activation, kernel_regularizer=l1_l2(l1=best_params['l1_reg'], l2=best_params['l2_reg'])))
            fnn.add(Dropout(best_params['dropout_rate'])) 
            fnn.add(Dense(1, activation='sigmoid'))

            # Compile the model with the optimizer
            fnn.compile(optimizer=best_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            model = fnn
            
            return model

    ##############################################################################################################

    # Call bounds for optimization 
    pbounds = get_hyperparameter_space()
    
    # Evaluate KNN model
    def evaluate_model(batch_size, epochs, learning_rate, dropout_rate, l1_reg, l2_reg):
        
        # Run FNN model
        fnn = Sequential()
        fnn.add(Dense(32, input_dim=X_train.shape[1], activation=best_activation, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        fnn.add(Dropout(dropout_rate)) 
        fnn.add(Dense(16, activation=best_activation, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        fnn.add(Dropout(dropout_rate))
        fnn.add(Dense(1, activation='sigmoid'))
    
        # Compile the model with the optimizer
        fnn.compile(optimizer=best_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model = fnn
        
        # Fit model
        model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.1, verbose=0)
    
        _, avg_score = fnn.evaluate(X_test, y_test, verbose=0)
        
        # Return the average accuracy
        return avg_score
    
    # Bayesian optimization
    best_params, optimizer, n_iter = bayesian_optimization(evaluate_model, pbounds, n_iter = 10)
    
    # Store the optimized model
    model = fnn_model(best_params)
    
    # Run the KNN model 
    model.fit(X_train, y_train, epochs=int(best_params['epochs']), batch_size=int(best_params['batch_size']), verbose=0)
    
    # Plot accuracy 
    model_name = "FNN"
    accuracy = plot_avg(optimizer, n_iter, X_test, y_test, model, model_name)
    
    model_type = "kernel"
    
    return model, accuracy, model_type, model_name
    
