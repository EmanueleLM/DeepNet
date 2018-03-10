# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:11:47 2018

@author: Emanuele

This module (distributed prediction -> dprediction) can be used to predict, 
    with a pool of neural networks some classification task
    - can be used with nets trained in parallel (or not, it's up to you);
    - you can specify the kind of classification task (majority vote, weighted
        majority vote, weighted average etc.).
This module must be easy to extend and transparent to all the other 
    (i.e. here go the neural nets and you have just to specify the
   kind of classification task, nothing more).
"""

import copy as cp
import numpy as np

# classify a sample based on weighted-majority vote of n nets
# takes as input:
#   x, the sample we need to classify
#   nets, a list of neural networks tranied on different samples (at least one)
# returns:
#   a vector of C entries, where C is the number of possible outcomes in the classification task
#   the index of the prediction out of C, from 0 (first class) to C-1 (last class)
def single_class_prediction(x, nets, weighted=False):
    
    net_importance = (1. if weighted==False else nets[0].accuracy_on_test);
    prediction = net_importance * cp.deepcopy(nets[0].net_activation(x));
    
    for n in range(len(nets)-1):  
        net_importance = (1. if weighted==False else nets[n].accuracy_on_test);
        prediction += net_importance * nets[n].net_activation(x);
        
    return prediction, np.argmax(prediction);

# classify a sample based on weighted-majority vote of n nets
# takes as input:
#   x, the sample we need to classify
#   nets, a list of neural networks tranied on different samples (at least one)
# returns:
#   the average of the prediction of each net, eventually weighted by their importance
def regression(x, nets, weighted=False):
    
    net_importance = (1. if weighted is False else nets[0].accuracy_on_test);
    prediction = net_importance * cp.deepcopy(nets[0].net_activation(x));
    
    for n in range(len(nets)-1):  
        net_importance = (1. if weighted is False else nets[n].accuracy_on_test);
        prediction += net_importance * nets[n].net_activation(x);
        
    return prediction/len(nets);