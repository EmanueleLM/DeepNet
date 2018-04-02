# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 23:18:05 2018

@author: Emanuele

Net generator module
"""

import numpy as np
import deepnet as dn
import genetic.crossover as cr
import genetic.fitness as fit
import genetic.mutation as mu
import genetic.selection as sel
import genetic.elitism as el

# dictionary for the crossover methods
dict_crossover = {"one-point": cr.one_point_crossover, 
                  "one-point-string": cr.one_point_crossover_string}; 
# dictionary for the mutations methods
dict_mutations = {"random": mu.random_mutation, 
                  "random-string": mu.random_mutation_string}; 
# dictionary for the selection methods
dict_selection = {"rank": sel.rank_selection, 
                  "roulette": sel.roulette_selection}; 

# dictionary of fitness functions
dict_fintess = {"accuracy": fit.classification_accuracy, 
                "kcomplexity": fit.simplicity };

# specify the min/max number of layers each layer can have
MAX_NUM_LAYERS = 6; 
MIN_NUM_LAYERS = 2; 

# specify the min/max number of neurons each layer can have
MAX_NUM_NEURONS = 20; 
MIN_NUM_NEURONS = 10; 

# generate a population of random neural networks
# takes as input:
#   i_size, dimensions of each input vector
#   o_size, dimension of each output vector
#   pop_size, the number of neural in the population
#   connection_percentage, percentage of connections that are active: a value of
#       1. means fully connected, while a value \in (0, 1) specifies the probability
#       that a connection for a given net is off
# returns:
#   net_population, a list that contains the nets of the fresh new population
def rand_population(i_size, o_size, pop_size, connection_percentage=1.):
    
    # list that contains the final population of nets
    net_population = list();
    
    # size of activations' dictionary in deepnet module
    dict_act_size = len(dn.activations_dict);
    # size of the losses' dictionary in deepnet module
    loss_dict_size = len(dn.loss_dict); 
    
    for n in range(pop_size):
        
        n_layers = np.random.randint(MIN_NUM_LAYERS, MAX_NUM_LAYERS);
        layer = np.array([]);
        
        # append a random layer
        for i in range(n_layers-1):            
            neurons = np.random.randint(MIN_NUM_NEURONS, MAX_NUM_NEURONS);
            activation = list(dn.activations_dict.keys())[np.random.randint(0, dict_act_size)];
            layer = np.append(layer, np.array([neurons, activation]));      
        
        # append the last layer
        activation = list(dn.activations_dict.keys())[np.random.randint(0, dict_act_size)];
        layer = np.append(layer, np.array([o_size, activation]));
        
        # generate the parameters for the net
        loss = list(dn.loss_dict.keys())[np.random.randint(0, loss_dict_size)];
        
        if connection_percentage == 1.:
            # tuple's form (verbose=, fully_connected=, connection_percentage=)
            params = (True, True, 1.);
            
        else:
            # tuple's form (verbose=, fully_connected=, connection_percentage=)
            params = (True, False, connection_percentage);
        
        # create the net and append it to the population
        net = dn.DeepNet(i_size, layer.reshape(n_layers, 2), loss, *params);
        net_population.append(net);
    
    return net_population;

# evaluate the fitness of a population of n nets
# takes as input the fitness function, which is put into a vocabulary 
# return a list [[net], fitness]^+
# takes as input:
#   population, a list that contains all the nets (not the couples <net, fitness>)
#   fitness, a function that returns the fitness for a given net
#   args, argumnets you pass to the fintess function
# returns:
#   a list of tuples <net, value_fitness>
def eval_fitness(nets, fitness, args=None):
    
    # <net, fitness> elements evaluation
    if args is None:
        population = list([[n, dict_fintess[fitness](n)] for n in nets]); 
    else:
        population = list([[n, dict_fintess[fitness](n, *(args))] for n in nets]);
    
    return population;

# evolve a population of nets
#   this is less 'blobby' than the previous version, still a mess: use this 
#    function just in case you want something that works fast, otherwise please
#    read the documentation and check the /test folder on how to setup an evolving 
#    envoirnment
# takes as input:
#   population_size, the number of nets to evolve
#   selection, a string that indicates the selection method used, taken by dict_selection
#   crossover, a string that indicates the crossover method used, taken by crossover_dict
#   mutation,  a string that indicates the mutation method used, taken by mutation_dict
#   p_crossover, the probability that two given nets, selected by a selection method, are subject to crossover
#   p_mutation, a probability that a net is subjecte to mutation (in each of its part)
# returns:
#   the population evolved with the genetic algorithm
def evolve(population, selection, crossover, mutation, p_crossover=.5, p_mutation=5e-2):
       
    for n in range(int(len(population)/2)):
        
        # select two nets and apply crossover
        id1, id2 = dict_selection[selection](population, ordered=True);
        p1, p2 = dict_crossover[crossover](population[id1], population[id2], p_crossover); 
        
        # apply mutation
        dict_mutations[mutation](p1, p_mutation);
        dict_mutations[mutation](p2, p_mutation);
        
    return population;    
    