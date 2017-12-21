# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:02:59 2017

@author: Emanuele

Activation functions, versions with single sample, with full dataset, both SIMD and batch.
"""

import numpy as np

# logistic function activation
def sigma(W, X, b):
    Z = np.dot(W.T,X) + b; # activate (linearly) the input
    return 1/(1+np.exp(-Z));
    
# ReLU activation function
def relu(W, X, b):
    Z = np.dot(W.T,X) + b; # activate (linearly) the input    
    Z[Z<=0] = 0;
    return Z;

# LeakyReLU activation function, where \epsilon is set to 0.01
def leakyRelu(W, X, b):
    Z = np.dot(W.T,X) + b; # activate (linearly) the input    
    Z[Z<=0] = 0.1;
    return Z;

# tanh activation function
def tanh(W, X, b):
    Z = np.dot(W.T,X) + b; # activate (linearly) the input
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z));

# exp activation function
def exp(W, X, b):
    Z = np.dot(W.T,X) + b; # activate (linearly) the input
    return np.exp(Z);

# linear activation function
def linear(W, X, b):
    Z = np.dot(W.T, X) + b; # activate (linearly) the input
    return Z;
