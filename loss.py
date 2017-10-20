# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:04:02 2017

@author: Emanuele

Loss functions for the backpropagation algorithm
"""

import numpy as np

# loss functions for the train set    
# respectively L2, L1 losses
# Y is the value predicted by our algorithm
# T is the real value (we are in a supervised scenario)
def lossL2(Y, T):
    return np.absolute(np.power((T-Y), 2));

def lossL1(Y, T):
    return np.abs((T-Y));

# loss function known as Cross Entropy
# Y is the value predicted by our algorithm
# T is the real value (we are in a supervised scenario)
# we expect that the number of dimensions is the first dimension of both the input (so Y.shape[1]=T.shape[1]=dimensions)
def lossCrossEntropy(T, Y):
    Y[Y==1] = 0.9999;
    Y[Y==0] = 0.0001
    return -T*np.log2(Y)-(1-T)*np.log2(1-Y);