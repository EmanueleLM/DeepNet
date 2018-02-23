# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:04:02 2017

@author: Emanuele

Loss functions for the backpropagation algorithm
"""

import numpy as np

# loss functions for the train set    
# respectively L2, L1 losses
# y is the value predicted by our algorithm
# t is the real value (we are in a supervised scenario)
def loss_L2(y, t):
    return np.absolute(np.power((t-y), 2));

def loss_L1(y, t):
    return np.abs((t-y));

# loss function known as Cross Entropy
# y is the value predicted by our algorithm
# t is the real value (we are in a supervised scenario)
# we expect that the number of dimensions is the first dimension of both the input (so y.shape[1]=t.shape[1]=dimensions)
def loss_cross_entropy(y, t):
    np.clip(y, 1e-10, 1-1e-10, out=y);
    return -t*np.log2(y)-(1-t)*np.log2(1-y);

# loss function known as Kullback-Leibler
# y is the value predicted by our algorithm
# t is the real value (we are in a supervised scenario)
# we expect that the number of dimensions is the first dimension of both the input (so y.shape[1]=t.shape[1]=dimensions)
# please note that KL can be written as entropy plus cross entropy of target/prediction
def loss_KL(y, t):
    np.clip(y, 1e-10, 1-1e-10, out=y);
    entropy = y*np.log2(y);
    cross_entropy = -t*np.log2(y)-(1-t)*np.log2(1-y);
    return cross_entropy - entropy;

# loss function known as Variational Autoencoders Loss (VAE)
# y is the value predicted by our algorithm
# t is the real value (we are in a supervised scenario)
# we expect that the number of dimensions is the first dimension of both the input (so y.shape[1]=t.shape[1]=dimensions)
# please note that VAE loss can be written as entropy of prediction minus kl of target/prediction
def loss_VAE(y, t):
    kl = loss_KL(y, t);
    np.clip(y, 1e-10, 1., out=y);
    y[y==0] = 1e-10;
    return np.log2(y) - kl;