# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:51:04 2017

@author: Emanuele

This module implements various mutations methods
"""

import deepnet as dn
import numpy as np

# increment/decrement of the percentage of neurons in a given layer, 
#   if mutation happens (e.g. .05 means +5% or -5%)
DELTA_CONNECTIONS = .05; 

# classic random mutation that happens with a low probability:
#    increment the number of neurons in a layer, for each layer;
#    change the activation function;
#    increment/decrement learning rate by a quantity that has the same magnitude 
#    of the learning rate of the net other things to be defined (change the dropout
#    rate, change the connections topology etc.).
# takes as input
#   net, the neural net coming from the specification in DeepNet module
#   p, the mutation's probability
# please note that if a change in the number of neurons for a given layer happens,
#    we change also the number of
#  weights and the associated matrix: we assume that the (possibly new) weights 
#   are distributed according to a unimodal 
#  distribution and we initialize them according to that, estimating the parameters
#    from the weights that already exist
# if we diminish the number of weights, we normalize the resulting ones according 
#   to mean and variance of the weights
#  of the original weights
def random_mutation(net, p):
    
    dict_act_size = len(dn.activations_dict); 
    dict_loss_size = len(dn.loss_dict);
    
    for l in range(net.activations.shape[0]-1):
        
        # change the activation function of the layer
        if np.random.rand() <= p:
            
            # we may have an assignment to the same activation function
            rand_activation = np.random.randint(0, dict_act_size-1);
            net.activations[l] = list(dn.activations_dict.keys())[rand_activation]; 
        
        # change the number of connections for a layer
        if np.random.rand() <= p:
            
            # calculate the number of new neurons in the connections as max between 1
            #   (we want to change something!) and the increment wrt the number
            #   of connections in the layer
            new_neurons = (1 if (np.random.rand()<=.5 or net.weights[l].shape[1]<=2) else -1)*max(1, int(DELTA_CONNECTIONS*net.weights[l].shape[1])); 
            
            # estimate mean and variance of the parameters before we change them
            mean_first_layer = np.mean(net.weights[l].flatten()); 
            var_first_layer = np.var(net.weights[l].flatten());
            mean_second_layer = np.mean(net.weights[l+1].flatten()); 
            var_second_layer = np.var(net.weights[l+1].flatten());
            # estimate the mean of the bias of the first layer
            mean_bias = np.mean(net.bias[l].flatten()); 
            var_bias = np.var(net.bias[l].flatten()); 
            
            #normalize the weights once we have removed/added some of them
            if new_neurons >= 0:
                
                norm_weights = np.random.normal(mean_first_layer, var_first_layer, 
                                                [net.weights[l].shape[0], new_neurons]);
                                                
                net.weights[l] = np.append(net.weights[l], norm_weights, axis=1);
                
                norm_weights = np.random.normal(mean_second_layer, var_second_layer, 
                                                [new_neurons, net.weights[l+1].shape[1]]);
                                                
                net.weights[l+1] = np.append(net.weights[l+1], norm_weights, axis=0);
                
                net.bias[l] = np.append(net.bias[l], np.random.normal(mean_bias, 
                        var_bias, [new_neurons, 1]), axis=0);
                
                # case of non fully connected net, insertion
                if net.fully_connected is False:
                    
                    net.mask[l] = np.append(net.mask[l], np.random.choice([0,1], 
                            size=[net.mask[l].shape[0], new_neurons], 
                            p=[1-net.connection_percentage, net.connection_percentage]), 
                    axis=1);
                    
                    net.mask[l+1] = np.append(net.mask[l+1], np.random.choice([0,1], 
                            size=[new_neurons, net.mask[l+1].shape[1]], 
                            p=[1-net.connection_percentage, net.connection_percentage]), 
                    axis=0);
            
            else:
                
                # remove the first new_neurons columns from the layer
                net.weights[l] = np.delete(net.weights[l], range(np.abs(new_neurons)), axis=1); 
                
                # remove the first new_neurons rows from the next leayer
                net.weights[l+1] = np.delete(net.weights[l+1], range(np.abs(new_neurons)), axis=0); 
                net.bias[l] = np.delete(net.bias[l], range(np.abs(new_neurons)), axis=0);
                
                # case of non fully connected net, deletion
                if net.fully_connected is False:
                    
                    # remove the first new_neurons columns from the layer
                    net.mask[l] = np.delete(net.mask[l], range(np.abs(new_neurons)), axis=1); 
                    
                    # remove the first new_neurons rows from the next leaye
                    net.mask[l+1] = np.delete(net.mask[l+1], range(np.abs(new_neurons)), axis=0); 
       
        # change the loss of the net
        if np.random.rand() <= p:
            
            net.loss = list(dn.loss_dict.keys())[np.random.randint(0, dict_loss_size-1)]; 
            
        # modify the learning rate of the net
        if np.random.rand() <= p:
            
            # half or double the learning rate with a probability of 0.5 each
            net.learning_rate += ((-1)**np.random.randint(0,2))*(10**np.log10(net.learning_rate))*(.5); 
        
        # modify the connectivity of the net, we modify each single connection with probability p
        if net.fully_connected is False:
            
            for m in net.mask:

                # -1 stands for eliminate connections, 0 none, 1 add connection
                m = np.add(m, np.random.choice([-1,0,1], size=m.shape, p=[p/2, 1-p, p/2]));
                
                # if we cancel a non existing connection we get a -1, 
                #   so we go back to a zero in the mask
                # if we add a connection to an exisitng one, we get a 2, 
                #   so we get back to 1 in the mask
                m[m<0] = 0; 
                m[m>0] = 1;
                         
# random mutation with strings that represents each net:
#    increment the number of neurons in a layer, for each layer;
#    change the activation function;
#    increment/decrement learning rate by a quantity that has the same magnitude 
#    of the learning rate of the net other things to be defined (change the dropout
#    rate, change the connections topology etc.).
# takes as input
#   net, the string that represent the encoded neural net
#   p, the mutation's probability
def random_mutation_string(net, p):
    
    dict_act_size = len(dn.activations_dict); 
    dict_loss_size = len(dn.loss_dict);
    
    new_net = '';
    sep = ',';
    
    # get each element of the net by splitting the string that represents it
    #  in its components
    el = net.split('[');
    input_size = int(el[0].replace(',', ''));
    
    # get the layers' description
    el = el[1].split(']');
    net_layers = el[0].split(',');
    
    # get the loss and learning rate
    el = el[1].split(',')[1:];
    net_loss = el[0].replace(',', '');
    net_l_rate = el[1].replace(',', '');
    
    # add the input
    new_net += str(input_size) + sep + '[';
    
    # mutate each parameter with a low probability
    for i in range(0,len(net_layers)-1,2):
        
        # change the number of connections for a layer
        if np.random.rand() <= p:
            
            # calculate the number of new neurons in the connections as max between 1
            #   (we want to change something!) and the increment wrt the number
            #   of connections in the layer
            net_layers[i] = int(net_layers[i]);
            net_layers[i] += (1 if (np.random.rand()<=.5 or net_layers[i]<=2) else -1)*max(1, int(DELTA_CONNECTIONS*net_layers[i])); 
        
        init_comma = ('' if i==0 else ',');
        new_net += init_comma + str(net_layers[i]) + sep;
                  
        # change the activation function of the layer
        if np.random.rand() <= p:
            
            # we may have an assignment to the same activation function
            rand_activation = np.random.randint(0, dict_act_size-1);
            net_layers[i+1] = list(dn.activations_dict.keys())[rand_activation];

        new_net += net_layers[i+1];
            
    new_net += ']';
        
    # change the loss of the net
    if np.random.rand() <= p:
        
        net_loss = list(dn.loss_dict.keys())[np.random.randint(0, dict_loss_size-1)]; 
    
    new_net += sep + net_loss;
            
    # modify the learning rate of the net
    if np.random.rand() <= p:
        
        net_l_rate = float(net_l_rate);
        # half or double the learning rate with a probability of 0.5 each
        net_l_rate += ((-1)**np.random.randint(0,2))*(10**np.log10(net_l_rate))*(.5); 
    
    new_net += sep + str(net_l_rate);
    
    # compress the string by eliminating usless blank spaces
    new_net = new_net.replace(' ', '');
            
    return new_net;