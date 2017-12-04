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
# returns
#   net1, net2, the modified nets after the crossover
def onePointCrossover(net1, net2, p):
    # crossover fails
    if np.random.rand() > p:
        return net1, net2;
    # mixing two nets that are shallow, we cannot mix completely what comes after the input, otherwise we would have no changes!
    # we just change the hidden layer and the loss function, and we leave unchanged the output layer (with its initial activation function)
    if net1.activations.shape[0] == net2.activations.shape[0] == 2:
        layer1 = np.array([net2.W[0].shape[1], net2.activations[0], net1.W[1].shape[1], net1.activations[1]]);
        layer2 = np.array([net1.W[0].shape[1], net1.activations[0], net2.W[1].shape[1], net2.activations[1]]);
        temploss = net1.loss; # this one is used to prevent that the second net has the same loss function as the first one
        net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net2.loss, False);
        net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, False);
    else: 
        # choose the cut point for both the networks, in this case we assume that half is a good split point for a one point crossover    
        cut_point = (np.ceil(net1.activations.shape[0]/2).astype('int'), np.ceil(net2.activations.shape[0]/2).astype('int'));
        # define the structure that is used to create the first net's layers as a composition of the first half of the first net plus the second half of the second
        layer1 = np.array([[net1.W[i].shape[1], net1.activations[i]] for i in range(cut_point[0])]);
        layer1 = np.append(layer1, np.array([[net2.W[i].shape[1], net2.activations[i]] for i in range(cut_point[1], net2.activations.shape[0])]));
        # define the structure that is used to create the second net's layers as a composition of the first half of the second net plus the first half of the second
        layer2 = np.array([[net2.W[i].shape[1], net2.activations[i]] for i in range(cut_point[1])]);
        layer2 = np.append(layer2, np.array([[net1.W[i].shape[1], net1.activations[i]] for i in range(cut_point[0], net1.activations.shape[0])]));
        # create the nets invoking the DeepNet module's __init__   
        temploss = net1.loss; # this one is used to prevent that the second net has the same loss function as the first one
        net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net1.loss, False);
        net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, False); 
    # swap the learning rates between the two nets
    temp = net1.learning_rate;
    net1.learning_rate = net2.learning_rate;
    net2.learning_rate = temp;
    return net1, net2;

# uniform crossover: takes two nets and mix them uniformely (i.e. each layer of net1 has 50% of probability to become part of
#  of net2, and viceversa).
# takes as input
#   net1, the first net (DeepNet object)
#   net2, the second net (DeepNet object)
#   p, the probability that the crossover happens (this one is usually much bigger than the mutation  probability, e.g. 50%)
# returns
#   net1, net2, the modified nets after the crossover
def uniformCrossover(net1, net2, p):
    pass;    


