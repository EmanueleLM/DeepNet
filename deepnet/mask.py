# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:50:27 2017

@author: Emanuele

This module handles a particular object we could name 'connections handler': its aim is to specify
 how the net is connected, in particular we distinguish between fully connected and partially connected nets.
In this sense we just use some binary 'masks' that are applied on the weights/biases of the net and that
 turn off the connections on that layer.
"""

import numpy as np
import re # regexp

# This class identifies the connections that each layer has i.e. the topology
#    of the net itself
#  we use a human way to define the connections, and a parser to obtain the 
#    informations to pass from the
#  human description to a computer like one
# This is how we define the human-like encoding of the connections of the net
#    with a (non-exactly correct but intuitive*) reg-exp:
#   [layer(i)':' [input(,i)^* '|' neuron(i+1)]^j]^n
#       layer(i) is the i.th layer out of the n in the net;
#       input(,i) represents the generic input (aka synapse) in the layer i;
#       neuron(i+1) is the neuron where the input(,i) comes into;
#       j is the j-th neuron in the layer i+1.
#   example: given a two layers net with 
   #    # of inputs I=10, # of hidden neurons J=15, # of outputs P=8
#            we want to specify that:
#               layer1: the firts 5 inputs connected to neurons from 1 go to 10
#               layer1: the last 5 inputs connected to neurons from 11 go to 15
#               layer2: the first 3 inputs are connected to the first 2 outputs
#               layer2: the 4th input is connected to the outputs from 3 to 5
#               layer2: the inputs from 5 to 15 connected to outputs from 6 to 8
#   'layer(1): 1:5|1:10, 6:10|11:15 layer(2): 1:3|1:2, 4|3:5, 5:15|6:8' 
#
#*: in the regexp there's an error which is intentional, 
#   [input(,i)^* '|' neuron(i+1)]^j should be [input(,i)^* '|' neuron(i+1) ',']^j-1 [input(,i)^* '|' neuron(i+1)]^1
#   if you devise other errors, please let me know
class Mask(object):
    
    # given a definition of connections, returns the binary matrices 
    #   that specify those connection
    # takes as input:
    #   net, the neural net used to extract the weights/biases shapes
    #   net_topology, the string specification of the connections of the net
    def __init__(self, net, net_topology):
        
        # create some 'masks' of weights as binary matrices that are used to put
        #   to zero the desired connections
        # assume a fully connected net
        self.weights = list();
        self.bias = list();
        
        for w in net.weights:
            self.weights.append(np.zeros(w.shape));
        self.create_mask(net_topology);
        
        for i in range(len(self.layers)):
            
            for j in range(len(self.layers[i])):
                
                 # split the range in two parts, e.g. '2:3|4:10' -> '2:3', '4:10'
                part1, part2 = re.split('\|', self.layers[i][j]);
                
                # process the two parts distinctly
                # consider the connections of the incoming layer,
                #   e.g. '3:5' -> 'range(3,5)'
                part1 = self.parse_range(part1, self.weights[i].shape[0]); 
                
                # really the same as above, but wrt the other layer
                part2 = self.parse_range(part2, self.weights[i].shape[1]); 
                part1 = eval(part1);
                part2 = eval(part2);
                
                # create the connection
                self.weights[i][min(part1)-1:max(part1),min(part2)-1:max(part2)] = 1; 
            
    # this function parses the input of the 'user' that wants to define a non-fully connected neural net
    def create_mask(self, net_topology):
        
        # parse the net_topology given by the input
        # remove all withespaces
        self.layers = re.sub(r"\s+", "", net_topology, flags=re.UNICODE); 
        # eliminate the header of each layer specification
        self.layers = re.split('layer\(\d+\):', self.layers)[1:]; 
        # return the sublists fo each layer descriptor
        self.layers = list([re.split(',', el) for el in self.layers]); 
        
    # this function takes as input a connection and returns one or more columns
    #   of the binary matrix used to define the topology of the net
    # takes as input
    #   connection, a connection in the form a:b|c:d (or a|b:c, a|b, a:b|c)
    def parse_connection(self, connection):
        
        return connection.replace('|', ',');
    
    # this function is used to parse the single connection range, mainly because 
    #   something like 'a:b' is not allowed in eval()
    #  so we just take 'a:b' (which can be also just 'a') and transform it 
    #   in a range that can be processed by eval 
    # takes as input:
    #   chunk, which is the string that indicated the range we wanto to specify
    #       the topology of, e.g. 'a:b'
    #   layer_shape, which is the integer that indicates the shape of layer of 
    #       the net we are dealing with and is used when the second term in the
    #       expression is not specified and we have something like 'a:' 
    #       (which is a shortcut of 'a:tot_num_parameters')
    def parse_range(self, chunk, layer_shape):
        
        # case 'a:b' (all the neruons from a-th to b-th, included)
        if re.match('(\d+):(\d+)', chunk): 
            r1, r2 = re.match('(\d+):(\d+)', chunk).group(1,2);
        # case 'a:' (all the neurons from a-th to the end of the layer, included)
        elif re.match('(\d+):', chunk):
            r1 = re.match('(\d+)', chunk).group(1);
            r2 = layer_shape;
        # case ':b' (all the neurons from the first to the b-th, included)
        elif re.match(':(\d+)', chunk):
            r2 = re.match(':(\d+)', chunk).group(1);
            r1 = 1;
        # case 'a' (just the neuron a-th)
        elif re.match('(\d+)', chunk):
            r2 = re.match('(\d+)', chunk).group(1);
            r1 = int(r2);
        # case ':' (all the neurons in the layer)
        elif re.match(':', chunk):
            r1 = 1;
            r2 = layer_shape;
        else:
            print('error while parsing string', chunk);
            return;
            
        return '['+str(r1) + ',' + str(r2) +']';         

""" Test """
verbose = False;

if verbose:
    
    example_str = 'layer(1): 1|1, 2|2:3, 3|4:5 layer(2): :|:';
    
    import deepnet as dn
    
    net = dn.DeepNet(3, np.array([[5, "tanh"], [6, "linear"]]), "L2", True); 
    mask = Mask(net, example_str);
    net.fully_connected = False;
    net.set_mask(mask.weights);
    
    import deepplot.netplot as nep
    
    nep.NetPlot(net);