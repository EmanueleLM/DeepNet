# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:50:14 2017

@author: Emanuele

This module implements various crossover methods
"""

import numpy as np
import deepnet as dn

# classical one point crossover: takes the two nets and divide them in two parts
# each first part is attached to the second part of the other net and we obtain two new nets
# obviously we need to modify (in general) the number of connections of the two new nets in the point
# where the original one are splitted (and eventually re-normalize the weights in those points)
# takes as input
#   net1, the first net (DeepNet object)
#   net2, the second net (DeepNet object)
#   p, the probability that the crossover happens (this one is usually much bigger than the mutation  probability, e.g. 50%)
def onePointCrossover(net1, net2, p):
    # scelgo a 'random' il punto di split
    # incrocio le due prime parti con le due seconde
    # aggiusto i pesi e i bias (avrò ragionevolmente delle dimensioni che mismatchano)
    if net1.activations.shape[0] == net2.activations.shape[0] == 2:
        return; # if we have two layers for both the net, we cannot mix them in a consistent way
    cut_point = (np.ceil(net1.activations.shape[0]/2).astype('int'), np.ceil(net2.activations.shape[0]/2).astype('int'));
    layers1_new = np.array([[net1.W[i].shape[0], net1.activations[i]] for i in range(cut_point[0])]);
    layers1_new = np.append(layers1_new,  np.array([[net2.W[j].shape[0], net2.activations[j]] for j in range(cut_point[1], net2.activations.shape[0])]));
    
    layers2_new = np.array([[net2.W[i].shape[0], net2.activations[i]] for i in range(cut_point[1])]);
    layers2_new = np.append(layers2_new,  np.array([[net1.W[j].shape[0], net1.activations[j]] for j in range(cut_point[0], net1.activations.shape[0])]));
    
    net1 = dn.DeepNet(net1.W[0].shape[0], layers1_new.reshape(int(layers1_new.shape[0]/2), 2), net1.loss);
    net2 = dn.DeepNet(net2.W[0].shape[0], layers2_new.reshape(int(layers2_new.shape[0]/2), 2), net2.loss);
    
    print(net1.activations, cut_point[0]);
    print(net2.activations, cut_point[1]);
    print(layers1_new, layers2_new); 
    
    
    