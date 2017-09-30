# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:08:35 2017

@author: Emanuele

Module that initializes the weights for the network wrt to different approaches (random, uniform, lecun etc.)
"""

import numpy as np

# function that returns a weights' numpy array initialized to random values between 0 and 1
def randomWeights(W):
    return np.random.randint(0, 101, W.shape)/100; # use broadcasting and this version of randint because it's easier to manage the shape of individual weight's vectors

# function that returns a weights' numpy array initialized to uniform values between upper and lower
# takes as input
#   W, the weights to fill with uniform values
#   lower, upper, rispectively the min a and max values on whose the uniform distribution of the weights is defined
def uniformWeights(W, lower, upper):
    return np.random.uniform(lower, upper, W.shape);

# initialize the weights in this way:
# pick up the weight's value from a uniform distribution between [-I,I]
#   where I is the multiplicative inverse of the number of connections to the unit
# If a neuron has 10 incoming synapses, it will recieve a weight from [-1/10,+1/10], uniformely distributed
# CAVEAT: please note that the weights are supposed to be submitted in such a way that the column's dimension is the number of incoming connections
def lecunWeights(W):
    I = 1/W.shape[0]; # number of incoming conncetions in the neuron
    return np.random.uniform(-I, I, W.shape);