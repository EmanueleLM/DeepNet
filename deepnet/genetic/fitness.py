# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 09:54:24 2018

@author: Emanuele

Fitness functions' module.
"""
# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");

import numpy as np
import utils.encoding as enc

# check how many predictions are correct
# takes as input:
#   Y_hat, the predictions of the net on a set of samples
#   T, the true prediction on the same samples
# returns:
#   the accuracy of the net
def classification_accuracy(Y_hat, T):
    
    # (number of corrects/number of samples) ratio
    n_samples = T.shape[-1];
    n_correct = 0.;
    
    for (y, t) in (Y_hat, T):
        
        if np.argmax(y) == np.argmax(t):
            n_correct += 1.;
    
    return (n_correct/n_samples);

# calculate the net's "simplicity" as the number of bits needed to encode it
def simplicity(net):
    pass;