# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:02:59 2017

@author: Emanuele

Activation functions, versions with single sample, with full dataset, both SIMD and batch.
"""

import numpy as np

# logistic function activation
def sigma(Z):
    return 1/(1+np.exp(-Z));
    
# ReLU activation function
def relu(Z):
    Z[Z<=0] = 0;
    return Z;

# LeakyReLU activation function, where \epsilon is set to 0.01
def leakyRelu(Z):
    Z[Z<=0] = 0.01;
    return Z;

# tanh activation function
def tanh(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z));