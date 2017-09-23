# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:59:16 2017 

@author: Emanuele

Deep network learner with parameter tuning and all sort of optimizations

The starting point is just a feed forward neural net where we add much more layers in depth.
"""

import numpy as np
import activations as act
import pylab as pl
import time

# we use a dictionary to handle each layer's activation function: we will need just to know the info contained in
#   DeepNet.layers to invoke the right function!
# Use this struct in this way: 
#   define a function in activations.py (imported let's say as act), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": act.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)
activations_dict = {"relu": act.relu, "sigmoid": act.sigma, "tanh": act.tanh};

# =============================================================================
#  class that models a deep network with multiple layers and different acrivation functions
#  take as input (the __init__ function):
#      input_size: size of a generic input sample
#      layers: a numpy.array (not a list since we will use numpy.size on it) that contains a number and a string: the former specifies the number of neurons on that layer
#          the latter specifies the activation function (e.g. a valid layers is list([[1,"relu"][5, "sigmoid"]]))
# =============================================================================
class DeepNet(object):
    def __init__(self, input_size, layers):
        self.W = list(); # list that contains all the weights in the net (we use a list and np.array for each matrix of weights, for efficiency reasons)
        self.W.append(np.array(np.zeros([input_size, np.int(layers[0][0])])));
        self.Bias = np.zeros([layers.shape[0], 1]);
        self.activations = np.array(layers[:,1]);
        for l in range(len(layers)-1):
            self.W.append(np.array(np.zeros([np.int(layers[l][0]), np.int(layers[l+1][0])])));
        print("\nNetwork created succesfully!")
        self.explainNet();
            
    # function that explains the net: this means that it describes the network
    # in terms of input, layers and activation functions
    def explainNet(self):
        print("\nThe network has ", self.W[0].shape[0], " input(s) and ", len(self.W), " layers.");
        for l in range(len(self.W)):
            print("The layer number ", l+1, " has ", self.W[l].shape[0], " input(s) per neuron, ", self.W[l].shape[1], " neuron(s) and ", self.activations[l], " as activation function.");
         
    # function that activates a given layer and based on its activation function, returns the desired output
    # takes as input
    #   the input X as a vector
    #   the layer where we want the activation to happen
    # returns
    #   the matrix of the activations (even one element for example in single output nn)
    def activation(self, X, layer):
        Z = np.dot(self.W[layer].T,X)+self.Bias[layer]; # activate (linearly) the input
        return activations_dict[self.activations[layer]](Z); # activate the actiovation function of each layer using the vocabulary defined at the beginning            
    
    # function that activates a all the layers in the net and returns the desired output
    # takes as input
    #   the input X as a vector
    # returns
    #   the output vector (even one element for example in single output nn)    
    def netActivation(self, X):
        res = X;
        for l in range(len(self.W)):
            res = self.activation(res, l);
        return res; 
    
""" Test part """

net = DeepNet(2, np.array([[3, "relu"],[5, "relu"], [7, "relu"], [1, "sigmoid"]]));
