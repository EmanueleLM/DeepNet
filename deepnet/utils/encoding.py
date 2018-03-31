# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:30:11 2018

@author: Emanuele

Turn a neural network into a unique identifier: you may want ot use 
 this encoding to store ypu networks somewhere on disk and being able to fully 
 restore them from a file, or perform fast mutation and crossover, then create 
 the nets (using the decoding functions) and train them.
 
Different encodigs are possible:
 - a lossless one, where the "nets' space" is isomorph to the output;
 - lossy, with compression dependent on the nature of the net itself.
 
For each encoding the decoding is provided in the same module, for the sake of 
 simplicity.
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../");

import deepnet as dn
import numpy as np

# encoding from whose you can fully reconstruct the neural net
# takes as inpput:
#   population, the set of nets
# returns:
#   a vocabulary of strings, where each string is encoded with all the info
#    needed to recontruct them
# 10, np.array([[100, "relu"]]), "L2", verbose=False, fully_connected=True, connection_percentage = .5);
# TODO: masks and non-fully connected topologies
def lossless_encoding(population):
    
    # output dictonary
    net_dict = {};
    
    # separator for each element in the target string
    sep=',';
    
    # one net casting to list type
    if type(population) != type([]):
        population = [population];
        
    for i in range(len(population)):
        
        encoding = str(population[i].weights[0].shape[0]);
        
        # start the layers' encoding
        encoding += ',[';
        
        for l in range(len(population[i].activations)):
            
            # add the i-th layer shapes and activation
            # exp = [neurons ',' activation ','] [neurons ',' activations]^+ ||
            #       [neurons ',' activation ','] 
            init_comma = ('' if l==0 else ',');
            encoding += init_comma + str(population[i].weights[l].shape[1]) + sep;
            encoding += population[i].activations[l];
        
        # end the layers' encoding
        encoding += ']';
        
        # add the loss and connections' parameters
        encoding += sep + str(population[i].loss);
        
        # add the learning rate
        encoding += sep + str(population[i].learning_rate);
            
        # add the string-net to the vocabulary
        net_dict[i] = encoding;
        
    return net_dict;
        
# lossless_decoding
# from a vocabulary of strings encoded with lossless_econding, return a list of
#  fresh new nets
# takes as input:
#   population, a vocabulary of strings where each entry encodes a neural net
# returns:
# list of nets (DeepNet objects)
# TODO: masks and non-fully connected topologies
def lossless_decoding(population):

    # vocabulary of target neural nets
    nets = {};
    
    for i in range(len(population)):
        
        # get each element of the net by splitting the string that represent it
        #  in its components
        el = population[i].split('[');
        input_size = int(el[0].replace(',', ''));
        
        el = el[1].split(']');
        net_layers = el[0].split(',');
        
        el = el[1].split(',')[1:];
        net_loss = el[0].replace(',', '');
        net_l_rate = el[1].replace(',', '');
        
        # create the net
        layers = np.array([[int(net_layers[l]), net_layers[l+1]] for l in range(0,len(net_layers)-1,2)]);
        net = dn.DeepNet(input_size, layers, net_loss);
        net.learning_rate = net_l_rate;
        
        # add the net to the vocabulary
        nets[i] = net;
        
    return nets;
        
        
# TODO:        
# encoding from whose you cannot fully reconstruct the neural net  
#   population, the set of nets
# returns:
#    a "minimal" representation of the net
def lossy_encoding(population):
    pass;    

# TODO:
# save your nets on a file and restore them when you need
def save_nets_on_file(population):
    pass;
