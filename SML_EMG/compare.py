#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:43:35 2023

@author: ariasarch
"""

def calculate_metrics(tp, fn, fp, tn):
    # Calculate Type I Error (False Positive Rate)
    type_1_error = fp / (fp + tn)

    # Calculate Type II Error (False Negative Rate)
    type_2_error = fn / (fn + tp)

    # Calculate Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn)

    # Calculate Specificity (True Negative Rate)
    specificity = tn / (tn + fp)

    # Calculate Precision (Positive Predictive Value)
    precision = tp / (tp + fp)

    # Calculate False Omission Rate
    false_omission_rate = fn / (fn + tn)

    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate False Discovery Rate
    false_discovery_rate = fp / (fp + tp)

    # Calculate Negative Predictive Value
    negative_predictive_value = tn / (tn + fn)

    return {
        'Type I Error': type_1_error,
        'Type II Error': type_2_error,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'False Omission Rate': false_omission_rate,
        'Accuracy': accuracy,
        'False Discovery Rate': false_discovery_rate,
        'Negative Predictive Value': negative_predictive_value,
    }