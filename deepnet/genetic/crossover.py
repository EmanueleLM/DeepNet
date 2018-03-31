# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:50:14 2017

@author: Emanuele

This module implements various crossover methods
"""

import numpy as np
import copy as cp
from itertools import chain

# classical one point crossover: takes the two nets and divide them in two parts
# each first part is attached to the second part of the other net and we obtain 
#   two new nets
# obviously we need to modify (in general) the number of connections of the two
#    new nets in the point
# where the original one are splitted (and eventually re-normalize the weights 
#   in those points)
# takes as input
#   net1, the first net (DeepNet object)
#   net2, the second net (DeepNet object)
#   p, the probability that the crossover happens (this one is usually much 
#       bigger than the mutation  probability, e.g. 50%)
# returns
#   net1, net2, the modified nets after the crossover
def one_point_crossover(net1, net2, p):
    
    # crossover fails
    if np.random.rand() > p:
        return net1, net2; # return the initial nets
    
    # choose the cut point for both the networks, in this case we assume that 
    #   half is a good split point for a one point crossovehalf is a good split
    #   point for a one point crossoverhalf is a good split point for a one point
    #   crossover   
    cut_1 = np.ceil(net1.activations.shape[0]/2).astype('int');
    cut_2 = np.ceil(net2.activations.shape[0]/2).astype('int');
    cut_point = (cut_1, cut_2);
    
    # copy the weights to transfer them
    W1 = cp.deepcopy(net1.weights);
    W2 = cp.deepcopy(net2.weights);
    
    # recreate them from scratch and re-assigm them to each net
    net1.weights = list();
    net2.weights = list();
    
    # assign the weights that we've saved
    net1.weights.append(W1[:cut_point[0]]);
    net1.weights.append(W2[cut_point[1]:]);
    net2.weights.append(W2[:cut_point[1]]);
    net2.weights.append(W1[cut_point[0]:]);    
    # flatten the lists
    net1.weights  = list(chain.from_iterable(net1.weights));
    net2.weights  = list(chain.from_iterable(net2.weights));
    
    # make the shapes of the weights coincide:
    #   (just at the cut points, we select to resize the new net accoring to 
    #   the minimum size of the two nets)
    if net1.weights[cut_point[0]-1].shape[1] > net1.weights[cut_point[0]].shape[0]:
        
        cut = np.s_[0:net1.weights[cut_point[0]-1].shape[1]-net1.weights[cut_point[0]].shape[0]];
        net1.weights[cut_point[0]-1] = np.delete(net1.weights[cut_point[0]-1], cut, axis=1);
        net1.bias[cut_point[0]-1] = np.delete(net1.bias[cut_point[0]-1], cut, axis=0);
        
    elif net1.weights[cut_point[0]-1].shape[1] < net1.weights[cut_point[0]].shape[0]:
        
        cut = np.s_[0:net1.weights[cut_point[0]].shape[0]-net1.weights[cut_point[0]-1].shape[1]];
        net1.weights[cut_point[0]] = np.delete(net1.weights[cut_point[0]], cut, axis=0);

    else:
        
        pass;
        
    if net2.weights[cut_point[1]-1].shape[1] > net2.weights[cut_point[1]].shape[0]:
        
        cut = np.s_[0:net2.weights[cut_point[1]-1].shape[1]-net2.weights[cut_point[1]].shape[0]];
        net2.weights[cut_point[1]-1] = np.delete(net2.weights[cut_point[1]-1], cut, axis=1);
        net2.bias[cut_point[1]-1] = np.delete(net2.bias[cut_point[1]-1], cut, axis=0);
        
    elif net2.weights[cut_point[1]-1].shape[1] < net2.weights[cut_point[1]].shape[0]:
        
        cut = np.s_[0:net2.weights[cut_point[1]].shape[0]-net2.weights[cut_point[1]-1].shape[1]];
        net2.weights[cut_point[1]] = np.delete(net2.weights[cut_point[1]], cut, axis=0);

    else:
        
        pass;
        
    # save the masks for the non-fully-connected topologies
    if (net1.fully_connected is False) and (net2.fully_connected is False): 
        
        mask1 = cp.deepcopy(net1.mask); 
        mask2 = cp.deepcopy(net2.mask);
        net_conn1 = net1.connection_percentage;
        net_conn2 = net2.connection_percentage;
        
        # recreate the masks
        net1.mask = list();
        net2.mask = list();        
        # assign the masks that we saved
        net1.mask.append(mask1[:cut_point[0]]);
        net1.mask.append(mask2[cut_point[1]:]);
        net2.mask.append(mask2[:cut_point[1]]);
        net2.mask.append(mask1[cut_point[0]:]);
        
        # flatten the lists
        net1.mask  = list(chain.from_iterable(net1.mask));
        net2.mask  = list(chain.from_iterable(net2.mask));
        # make the shapes of the masks coincide 
        #   (just at the cut points, we select to resize the new net accoring 
        #   to the minimum size of the two nets)
        if net1.mask[cut_point[0]-1].shape[1] > net1.mask[cut_point[0]].shape[0]:
            
            net1.mask[cut_point[0]-1] = np.delete(net1.mask[cut_point[0]-1], np.s_[0:net1.mask[cut_point[0]-1].shape[1]-net1.mask[cut_point[0]].shape[0]], axis=1);
        
        elif net1.mask[cut_point[0]-1].shape[1] < net1.mask[cut_point[0]].shape[0]:
            
            net1.mask[cut_point[0]] = np.delete(net1.mask[cut_point[0]], np.s_[0:net1.mask[cut_point[0]].shape[0]-net1.mask[cut_point[0]-1].shape[1]], axis=0);
       
        else:
            
            pass;
            
        if net2.mask[cut_point[1]-1].shape[1] > net2.mask[cut_point[1]].shape[0]:
            
            net2.mask[cut_point[1]-1] = np.delete(net2.mask[cut_point[1]-1], np.s_[0:net2.mask[cut_point[1]-1].shape[1]-net2.mask[cut_point[1]].shape[0]], axis=1);
        
        elif net2.mask[cut_point[1]-1].shape[1] < net2.mask[cut_point[1]].shape[0]:
            
            net2.mask[cut_point[1]] = np.delete(net2.mask[cut_point[1]], np.s_[0:net2.mask[cut_point[1]].shape[0]-net2.mask[cut_point[1]-1].shape[1]], axis=0);
        
        else:
            
            pass;
            
        # swap the connection percentages of each net
        net1.connection_percentage = net_conn1;
        net2.connection_percentage = net_conn2;
        
    # swap the losses between the nets
    temp = net1.loss;
    net1.loss = net2.loss;
    net2.loss = temp;
    
    # swap the learning rates between the two nets
    temp = net1.learning_rate;
    net1.learning_rate = net2.learning_rate;
    net2.learning_rate = temp;
    
    return net1, net2;

# TODO:
# one point crossover with strings that represents each net: takes the two nets
#  and divide them in two parts each first part is attached to the second part 
#  of the other net and we obtain two new nets
# obviously we need to modify (in general) the number of connections of the two
#    new nets in the point
# where the original one are splitted
# takes as input
#   net1, the first net (DeepNet object)
#   net2, the second net (DeepNet object)
#   p, the probability that the crossover happens (this one is usually much 
#       bigger than the mutation  probability, e.g. 50%)
# returns
#   net1, net2, the modified nets after the crossover
def one_point_crossover_string(net1, net2, p):
    pass;

# TODO:
# uniform crossover: takes two nets and mix them uniformely (i.e. each layer of
#    net1 has 50% of probability to become part of net2, and viceversa).
# takes as input
#   net1, the first net (DeepNet object)
#   net2, the second net (DeepNet object)
#   p, the probability that the crossover happens (this one is usually much 
#       bigger than the mutation  probability, e.g. 50%)
# returns
#   net1, net2, the modified nets after the crossover
def uniform_crossover(net1, net2, p):
    pass;    