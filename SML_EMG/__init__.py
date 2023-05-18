#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:39:54 2023

@author: ariasarch
"""

from SML_EMG.load_data import *
from SML_EMG.xgboost import *
from SML_EMG.LogitBoost import *
from SML_EMG.Adaboost import *
# from SML_EMG.LightGBM import *
from SML_EMG.decision_trees import *
from SML_EMG.linear_discriminant import *
from SML_EMG.quadratic_discriminant import *
from SML_EMG.nb_gaussian import *
from SML_EMG.nb_bernoulli import *
from SML_EMG.nb_multinomial import *
from SML_EMG.SVM_Linear import *
from SML_EMG.SVM_Quadratic import *
from SML_EMG.SVM_Cubic import *
from SML_EMG.SVM_Fine import *
from SML_EMG.SVM_Medium import *
from SML_EMG.SVM_Coarse import *
from SML_EMG.KNN_Fine import *
from SML_EMG.KNN_Medium import *
from SML_EMG.KNN_Coarse import *
from SML_EMG.KNN_Cubic import *
from SML_EMG.KNN_Weighted import *
from SML_EMG.KNN_Cosine import *
from SML_EMG.Random_Forest import *
from SML_EMG.Extra_Trees import *
from SML_EMG.fnn import *
from SML_EMG.SHAP import *
from SML_EMG.p_value import *
from SML_EMG.load_models import *

__version__ = '0.1.0'