# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:59:56 2017

@author: Emanuele

This module implements the selection methods: i.e. given a population of n nets
 it selects how to couple them according to a specific selection method 
 (e.g. rank selection, roulette wheel selection etc.)
"""

import numpy as np
from operator import itemgetter 


# the well known rank selection method to the population of nets
# takes as input
#    population, the population as tuples of <net, fitness>
#    ordered, boolean (that is considered True if not given) that means that 
#       the population is already ordered by its fitness in a descending order 
#       (i.e. the first net is the 'best' one)
# returns:
#   (id1, id2), a couple of integers (in the range of the population size,
#       which is passed by its reference) that inidcates the two nets to be 
#       coupled with crossover
# how it is implemented:
#   take a random number p: 
#       check wheter the number is greater than .5 (this happens 50% of the times)
#       if it is not true, re-iterate with .25 (25%) and so on..
#   the index of the iteration for which the disequality holds is id1
#   do it again for the second index, id2 
def rank_selection(population, ordered=True):
    
    # if we have a non-ordered population (by fitness), order it
    if ordered is False: 
        
        population = sort_by_fitness(population); 
    
    # draw two elements from the population    
    p = np.random.rand();
    
    for i in range(len(population)):
        
        if p >= 2**(-i-1):            
            id1 = i;
            break;
            
        if i == len(population)-1:
            id1 = i;
            
    p = np.random.rand();
    for i in range(len(population)):
        
        if p >= 2**(-i-1):
            id2 = i;
            break;
            
        if i == len(population)-1:
            id2 = i;
            
    # if by chanche we choose the same elements, we put together the one to the left
    #   (better wrt fitness)
    # may handle this situation differently, this solution is highly elitarian
    if id1 == id2:
        
        if id1 == 0:
            
            id2 = 1;
            
        else:
            
            id2 = id1-1;
            
    return (id1,id2);

# the well known roulette wheel selection
# takes as input
#   population, the population as tuples of <net, fitness>
#   ordered, boolean (that is considered True if not given) that means that 
#   the population is already ordered by its fitness in a descending order 
#   (i.e. the first net is the 'best' one)
# returns:
#   (id1, id2), a couple of integers (in the range of the population size, 
#   which is passed by its reference) that inidcates the two nets to be coupled with crossover
# how it is implemented:
#   assign to each net a probability that is proportional to its fitness
#   select nets according to the prob assigned to each member of the population
def roulette_selection(population, ordered=True):
    
    # if we have a non-ordered population (by fitness), order it
    if ordered is False: 
        
        population = sort_by_fitness(population); 
        
    total_fitness = np.sum(p[1] for p in population); 
    partial_fitness = np.array([f[1]/total_fitness for f in population]);
    
    # generate probability intervals for each individual
    probs = [np.sum(partial_fitness[:i+1]) for i in range(len(partial_fitness))];
    
    # draw two elements from the population
    for (i, individual) in enumerate(population):
        
        if np.random.rand() <= probs[i]:
            id1 = i;
            break;
    # ..      
    for (i, individual) in enumerate(population):
        if np.random.rand() <= probs[i]:
            id2 = i;
            break;
            
    # if by chanche we choose the same elements, we put together the one to the left
    #   (better wrt fitness)
    # may handle this situation differently, this solution is highly elitarian
    if id1 == id2:
        
        if id1 == 0:
            
            id2 = 1;
            
        else:
            
            id2 = id1-1;
            
    return (id1,id2);
       
# utility that orders the population of <net, fitness> in descending/ascendig order
#    wrt the fitness itself
# takes as input:
#       population, the population as tuples of <net, fitness>
#       desc, boolean (that is considered False if not given) that means that 
#       the resulting population will be in descending order, otherwise ascending
def sort_by_fitness(population, desc=False):
    
    return sorted(population, key = itemgetter(1), reverse=False);