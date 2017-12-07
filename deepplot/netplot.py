# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:13:02 2017

@author: Emanuele

This code is a modified version of visualise-neural-network at https://github.com/miloharper/visualise-neural-network
I just changed a bit the code in such a way that it is possible to visualize non-fully connected topologies
"""

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self, netmask=None, l=0):
        for i in range(len(self.neurons)):
            self.neurons[i].draw()
            if self.previous_layer:
                for j in range(len(self.previous_layer.neurons)):
                    if netmask is None and l>0:
                        self.__line_between_two_neurons(self.neurons[i], self.previous_layer.neurons[j]);
                    else:
                        if netmask[l-1][j][i] == 1:
                            self.__line_between_two_neurons(self.neurons[i], self.previous_layer.neurons[j]);


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self, netmask=None):
        for i in range(len(self.layers)):
            if netmask is not None:
                self.layers[i].draw(netmask, i);
            else:
                self.layers[i].draw(None);
        pyplot.axis('scaled');
        pyplot.show();

class NetPlot():
    def __init__(self, net):
        global vertical_distance_between_layers, horizontal_distance_between_neurons, neuron_radius, number_of_neurons_in_widest_layer;
        vertical_distance_between_layers = 10;
        horizontal_distance_between_neurons = 5;
        neuron_radius = 0.5
        number_of_neurons_in_widest_layer = np.max([net.W[i].shape[0] for i in range(len(net.W))]);
        network = NeuralNetwork();
        network.add_layer(net.W[0].shape[0]);
        for i in range(len(net.W)):
            network.add_layer(net.W[i].shape[1]);
        network.draw(None if net.fully_connected is True else net.mask);
        
""" Test """
verbose = False;
if verbose is True:
    import deepnet as dn # this one conflicts with deepnet module so make all this stuff 'un'-verbose before launching any other module
    net = dn.DeepNet(8, np.array([[10, "tanh"], [7, "leakyrelu"]]), "L2", True); # create the net
    # the first layer is divided in two regions connected, respectively, to each half of the second layer, while the second layer is fully connected to the output
    net.netTopology('layer(1): 1:3|1:2, 4:5|3, 6|4:9, 7:8|10 layer(2): :3|:7, 4:10|7'); 
    NetPlot(net);