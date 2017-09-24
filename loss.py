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
    return np.power((Y-T), 2);

def lossL1(Y, T):
    return np.abs((Y-T));

# loss function known as Cross Entropy
# Y is the value predicted by our algorithm
# T is the real value (we are in a supervised scenario)
def lossCrossEntropy(T, Y):
    if Y==T: # this if else is to avoid numerical problems with numpy.log2(0)
        return 0;
    elif (Y==1 and T==0) or (Y==0 and T==1):
        return 1;
    return -T*np.log2(Y)-(1-T)*np.log2(1-Y);