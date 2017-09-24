# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:16:55 2017

@author: Emanuele

This module calculates the various derivatives of well know functions, and is
used to perform backpropagation in the scenario where we do not know:
    how many activations functions (i.e. layers) we have in our net
    how they are ordered
"""

import numpy as np

# derivative of Y wrt the exit variable for the L2 loss function
# since we use the function (T-Y)^2, we obtain 2(T-Y) as derivative
def dYL2(Y, T):
    return 2*np.add(T, -Y);

# derivative of Y wrt the exit variable for the L1 loss function
def dYL1(Y, T):
    return;
    
# derivative of Y wrt the exit variable for the Cross Entropy loss function  
def dYCrossEntropy(Y, T):
    return np.add(T, -Y);