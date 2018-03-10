# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:28:20 2018

@author: Emanuele

This module (distributed train -> dtrain) provides the way to train a net on a 
    dataset: in particular you can:
        - dispatch this module multiple times for multiple parallel training.
"""

import numpy as np

# train a net on the train set;
# validate the net using the early stopping technique;
# test the net performances;
# takes as input:
#   net, the neural network we want to train
#   X, T the dataset samples + the desired output: they come in the shape 
#       (dimensions, num_of_samples) <- seach column a sample
#   epochs, the number of epochs we want to train the net
#   {train, validation, test}_percentage, the percentage of the dataset that is 
#       used for train, validation and testing, respectively
def train(net, X, T, epochs=10, train_percentage=.7, validation_percentage=.2, test_percentage=.1):
    
    # TODO:  split the dataset according to the percentages
    # train the net
    # TODO: validate (and eventually stop the learning)
    # TODO: test the perfomances and set net.accuracy_on_test according to the results
    for e in range(epochs):
        for n in range(X.shape[1]):
            net.backpropagation(X[:,n].reshape(net.weights[0].shape[0],1), T[:,n].reshape(net.weights[-1].shape[1],1));
    return net;