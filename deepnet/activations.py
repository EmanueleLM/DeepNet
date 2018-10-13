# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:02:59 2017

@author: Emanuele

Activation functions, versions with single sample, with full dataset, 
    both SIMD and batch.
"""

import numpy as np

# logistic function activation
def sigma(weights, X, bias):
    z = np.dot(weights.T,X) + bias; # activate (linearly) the input
    return 1/(1+np.exp(-z));
    
# ReLU activation function
def relu(weights, X, bias):
    z = np.dot(weights.T,X) + bias; # activate (linearly) the input   
    np.clip(z, 0., None, out=z);
    return z.astype(float);

# LeakyReLU activation function, where \epsilon is set to 1e-4
def leaky_relu(weights, X, bias):
    z = np.dot(weights.T,X) + bias; # activate (linearly) the input 
    z[z<0] *= 1e-4;
    return z.astype(float);

# tanh activation function
def tanh(weights, X, bias):
    z = np.dot(weights.T,X) + bias; # activate (linearly) the input
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z));

# exp activation function
def exp(weights, X, bias):
    z = np.dot(weights.T,X) + bias; # activate (linearly) the input
    return np.exp(z);

# linear activation function
def linear(weights, X, bias):
    z = np.dot(weights.T, X) + bias; # activate (linearly) the input
    return z;