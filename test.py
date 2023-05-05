#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:07:07 2023

@author: ariasarch
"""

import SML_EMG as se

X_train, X_test, y_train, y_test = se.load()
# funcs = [se.exec_xgboost, se.exec_logitboost]
funcs = [se.exec_xgboost]

dic = {}
for func in funcs:
    model, accuracy = func(X_train, X_test, y_train, y_test)
    dic[accuracy] = model
    
key = sorted(list(dic.keys()), reverse=True)[0]
best_model = dic[key]

se.shap_exp_tree(best_model, X_train, X_test)
