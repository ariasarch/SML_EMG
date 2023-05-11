#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:04:51 2023

@author: ariasarch
"""

import pickle

def load_models_and_data(funcs):
    
    # Load the model from disk
    loaded_models = {}
    for func in funcs:
        model_filename = func.__name__.replace("exec_", "") + '_model.sav'
        loaded_models[func.__name__.replace("exec_", "")] = pickle.load(open(model_filename, 'rb'))
    
    # Load the dic from the disk
    loaded_dic = pickle.load(open('dictionary.sav', 'rb'))
    
    # Load the SHAP from the disk
    loaded_SHAP = pickle.load(open('shap_values.sav', 'rb'))

    return loaded_models, loaded_dic, loaded_SHAP