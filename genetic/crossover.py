# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:50:14 2017

@author: Emanuele

This module implements various crossover methods
"""

import numpy as np
import deepnet as dn
import copy as cp

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
        # save the masks for the non fully connected topologies
        if (net1.fully_connected is False) and (net2.fully_connected is False): 
            mask1 = cp.deepcopy(net1.mask); # copy the mask of the first net (DeepNet __init__ destroys the mask components)
            mask2 = cp.deepcopy(net2.mask); # copy the mask of the second net
            net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net2.loss, verbose=False);
            net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, verbose=False);
            # explicitely assign the masks because the __init__ function of DeepNet would otherwise set them from zero and randomly 
            net1.setMask(mask2); # set the mask of the first net equals to the mask of the second net (a swap..)
            net2.setMask(mask1); # ..
            net1.setFullyConnected(False); # remember to set that the new net is not fully connected (ok maybe we have to change the __init__ of DeepNet to make it automatic :/ )
            net2.setFullyConnected(False); # ..
        # fully connected topologies' case
        else:
            net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net2.loss, False);
            net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, False);
    # case with two nets with numbers of layers different from 2, for both the nets    
    else: 
        """ TODO: make this part works with topologies non fully connected with possibily more than one hidden layer: a 'bit' of work is required """
        # choose the cut point for both the networks, in this case we assume that half is a good split point for a one point crossover    
        cut_point = (np.ceil(net1.activations.shape[0]/2).astype('int'), np.ceil(net2.activations.shape[0]/2).astype('int'));
        # define the structure that is used to create the first net's layers as a composition of the first half of the first net plus the second half of the second
        layer1 = np.array([[net1.W[i].shape[1], net1.activations[i]] for i in range(cut_point[0])]);
        layer1 = np.append(layer1, np.array([[net2.W[i].shape[1], net2.activations[i]] for i in range(cut_point[1], net2.activations.shape[0])]));
        # define the structure that is used to create the second net's layers as a composition of the first half of the second net plus the first half of the second
        layer2 = np.array([[net2.W[i].shape[1], net2.activations[i]] for i in range(cut_point[1])]);
        layer2 = np.append(layer2, np.array([[net1.W[i].shape[1], net1.activations[i]] for i in range(cut_point[0], net1.activations.shape[0])]));  
        # make a copy of the masks of both the nets
        if (net1.fully_connected is False) and (net2.fully_connected is False): 
            mask1 = cp.deepcopy(net1.mask);
            mask2 = cp.deepcopy(net2.mask);
            net_conn1 = net1.connection_percentage;
            net_conn2 = net2.connection_percentage;
            temploss = net1.loss; # this one is used to prevent that the second net has the same loss function as the first one
            net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net1.loss, False);
            net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, False);
            # assign the masks that we saved
            net1.mask.append(mask1[:cut_point[0]]);
            net1.mask.append(mask2[cut_point[1]:]);
            net2.mask.append(mask2[:cut_point[1]]);
            net2.mask.append(mask1[cut_point[0]:]);
            net1.mask  = [val for sublist in net1.mask for val in sublist];
            net2.mask  = [val for sublist in net2.mask for val in sublist];
            # make the shapes of the masks coincide (just at the cut points, we select to resize the new net accoring to the minimum size of the two nets)
            if net1.mask[cut_point[0]-1].shape[1] < net1.mask[cut_point[0]].shape[0]:
                net1.mask[cut_point[0]] = np.delete(net1.mask[cut_point[0]], net1.mask[cut_point[0]].shape[0]-net1.mask[cut_point[0]-1].shape[1], axis=0);
            else:
                net1.mask[cut_point[0]-1] = np.delete(net1.mask[cut_point[0]-1], net1.mask[cut_point[0]-1].shape[1]-net1.mask[cut_point[0]].shape[0], axis=1);
            if net2.mask[cut_point[1]-1].shape[1] < net2.mask[cut_point[1]].shape[0]:
                net2.mask[cut_point[1]] = np.delete(net2.mask[cut_point[1]], net2.mask[cut_point[1]].shape[0]-net2.mask[cut_point[1]-1].shape[1], axis=0);
            else:
                net2.mask[cut_point[1]-1] = np.delete(net2.mask[cut_point[1]-1], net2.mask[cut_point[1]-1].shape[1]-net2.mask[cut_point[1]].shape[0], axis=1);
            temploss = net1.loss; # this one is used to prevent that the second net has the same loss function as the first one
            net1 = dn.DeepNet(net1.W[0].shape[0], layer1.reshape(int(layer1.shape[0]/2), 2), net1.loss, verbose=False);
            net2 = dn.DeepNet(net2.W[0].shape[0], layer2.reshape(int(layer2.shape[0]/2), 2), temploss, verbose=False); 
            net1.mask = mask1;
            net2.mask = mask2;
            net1.fully_connected = False;
            net2.fully_connected = False;
            net1.connection_percentage = net_conn1;
            net2.connection_percentage = net_conn2;                
        else:
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