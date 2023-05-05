#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:41:24 2023

@author: ariasarch
"""

from setuptools import setup, find_packages

setup(
    name='emg_data_preprocessing',
    version='0.1.0',
    author='Ari Asarch',
    description='A package for preprocessing EMG data',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib'
    ],
)