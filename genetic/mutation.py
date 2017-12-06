# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:51:04 2017

@author: Emanuele

This module implements various mutations methods
"""

import deepnet as dn
import numpy as np

DELTA_CONNECTIONS = .05; # this parameter is the increment/decrement of the percentage of neurons in a given layer, if mutation happens (e.g. .05 menas +5% or -5%)


# classic random mutation
# with a low probability:
#    increment the number of neuron in a layer, for each layer;
#    change the activation function;
#    increment/decrement learning rate by a quantity that has the same magnitude of the learning rate of the net
#    other things to be defined (change the dropout rate, change the connections topology etc.).
# takes as input
#   net, the neural net coming from the specification in DeepNet module
#   p, the mutation's probability
# please note that if a change in the number of neurons for a given layer happens, we change also the number of
#  weights and the associated matrix: we assume that the (possibly new) weights are distributed according to a unimodal 
#  distribution and we initialize them according to that, estimating the parameters from the weights that already exist
# if we diminish the number of weights, we normalize the resulting ones according to mean and variance of the weights
#  of the original weights
def randomMutation(net, p):
    act_dict_size = len(dn.activations_dict); # size of the activations' dictionary
    loss_dict_size = len(dn.loss_dict); # size of the losses' dictionary
    for l in range(net.activations.shape[0]-1):
        # change the activation function of the layer
        if np.random.rand() <= p:
            net.activations[l] = list(dn.activations_dict.keys())[np.random.randint(0, act_dict_size-1)]; # we may have an assignment to the same activation function, it's something we know and accept
        # change the number of connections for a layer
        if np.random.rand() <= p:
            new_neurons = (1 if (np.random.rand()<=.5 or net.W[l].shape[1]<=2) else -1)*max(1, int(DELTA_CONNECTIONS*net.W[l].shape[1])); # calculate the number of new neurons in the connections as max between 1 (we want to change something!) and the incremen wrt the number of connections int he layer
            mean_first_layer = np.mean(net.W[l].flatten()); # estimate mean and variance of the weights before we change them
            var_first_layer = np.var(net.W[l].flatten()); # ..
            mean_second_layer = np.mean(net.W[l+1].flatten()); # estimate mean and variance of the next layer's weights before we change them
            var_second_layer = np.var(net.W[l+1].flatten()); # ..
            mean_bias = np.mean(net.Bias[l].flatten()); # estimate the mean of the bias of the first layer
            var_bias = np.var(net.Bias[l].flatten()); # estimate the variance of the bias of the first layer
            """ TODO(?): remember that maybe it's better to normalize the weights once we have removed/added """
            if new_neurons >= 0:
                net.W[l] = np.append(net.W[l], np.random.normal(mean_first_layer, var_first_layer, [net.W[l].shape[0], new_neurons]), axis=1);
                net.W[l+1] = np.append(net.W[l+1], np.random.normal(mean_second_layer, var_second_layer, [new_neurons, net.W[l+1].shape[1]]), axis=0);
                net.Bias[l] = np.append(net.Bias[l], np.random.normal(mean_bias, var_bias, [new_neurons, 1]), axis=0);
                # case of non fully connected net, insertion
                if net.fully_connected is False:
                    net.mask[l] = np.append(net.mask[l], np.random.choice([0,1], size=[net.mask[l].shape[0], new_neurons], p=[1-net.connection_percentage, net.connection_percentage]), axis=1);
                    net.mask[l+1] = np.append(net.mask[l+1], np.random.choice([0,1], size=[new_neurons, net.mask[l+1].shape[1]], p=[1-net.connection_percentage, net.connection_percentage]), axis=0);
            else:
                net.W[l] = np.delete(net.W[l], range(np.abs(new_neurons)), axis=1); # remove the first new_neurons columns from the layer
                net.W[l+1] = np.delete(net.W[l+1], range(np.abs(new_neurons)), axis=0); # remove the first new_neurons rows from the next leayer
                net.Bias[l] = np.delete(net.Bias[l], range(np.abs(new_neurons)), axis=0);
                # case of non fully connected net, deletion
                if net.fully_connected is False:
                    net.mask[l] = np.delete(net.mask[l], range(np.abs(new_neurons)), axis=1); # remove the first new_neurons columns from the layer
                    net.mask[l+1] = np.delete(net.mask[l+1], range(np.abs(new_neurons)), axis=0); # remove the first new_neurons rows from the next leayer
        # change the loss of the net
        if np.random.rand() <= p:
            net.loss = list(dn.loss_dict.keys())[np.random.randint(0, loss_dict_size-1)]; 
        # modify the learning rate of the net
        if np.random.rand() <= p:
            net.learning_rate += ((-1)**np.random.randint(0,2))*(10**np.log10(net.learning_rate))*(.5);
        # modify the connectivity of the net, we modify each single connection with probability p
        if net.fully_connected is False:
            for m in net.mask:
                m = np.add(m, np.random.choice([-1,0,1], size=m.shape, p=[p/2, 1-p, p/2])); # -1 stands for eliminate connections, 0 none, 1 add connection
                m[m<0] = 0; # if we cancel a non existing connection we get a -1, so we go back to a zero in the mask
                m[m>0] = 1; # if we add a connection to an exisitng one, we get a 2, so we get back to 1 in the mask
                
            