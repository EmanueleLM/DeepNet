# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:59:56 2017

@author: Emanuele

This module implements the selection methods: i.e. given a population of n nets
it selects how to couple them according to a specific selection method (e.g. rank selection, roulette wheel selection etc.)
"""

import numpy as np
from operator import itemgetter # used to sort the vector of population <net, fitness> by fitness


# this function applies the well known rank selection method to the population of nets
# takes as input
#    population, the population as tuples of <net, fitness>
#    ordered, boolean (that is considered True if not given) that means that the population is already ordered by its fitness in a descending order (i.e. the first net is the 'best' one)
# returns:
#   a couple of integers (in the range of the population size, which is passed by its reference) that inidcates the two nets to be coupled with crossover
# how it is implemented:
#   take a random number p: 
#       check wheter the number is greater than .5 (this happens 50% of the times)
#       if it is not true, re-iterate with .25 (25%) and so on..
#   the index of the iteration for which the disequality holds is id1
#   do it again for the second index, id2 
def rankSelection(population, ordered=True):
    if ordered is False: # if we have a non-ordered population (by fitness), order it
        population = sortByFitness(population); 
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
    if id1 == id2:
        if id1 == 0:
            id2 = 1;
        else:
            id2 = id1-1;
    return (id1,id2);

# utility that orders the population of <net, fitness> in descending/ascendig order wrt the fitness itself
# takes as input:
#        population, the population as tuples of <net, fitness>
#        desc, boolean (that is considered True if not given) that means that the resulting population will be in descending order, otherwise ascending
def sortByFitness(population, desc=True):
    return sorted(population, key = itemgetter(1), reverse=True);