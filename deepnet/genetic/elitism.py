# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:20:22 2017

@author: Emanuele

This module implements elitism techniques, in order to have a monotone improvement 
 of the fitness of our population: elitism basically predicts to save, 
 at the beginning of each crossover/mutation stage, the best k elements and to 
 preserve them into the population.
This module implements both the elitism selection and the insertion into the new
 population.
"""

import copy as cp 
from operator import itemgetter

# select a number of elements to preserve against mutation and crossover
#  first of all it orders the population by fitness (if needed, otherwise skip)
# takes as input:
#    population, the couples <net, fitness>
#    elite_size, the number of element to preserve against crossover and mutation
#    ordered, a boolean (set to True) that indicates if the population list is 
#       ordered by fitness (desc)
# returns
#    the vector of best nets
def elitism(population, elite_size, ordered=False):
    
    if ordered is False:        
        population = sort_by_fitness(population);
        
    if elite_size >= len(population):        
        return population, population;
    
    else:        
        return cp.deepcopy(population[:elite_size]), population;
    
# test whether the elite is 'robust' by letting all the members of the elite compete 
#  on all the misclassified examples, with a learning rate that is considerably 
#  bigger than the one of each net and is adjusted by considering how many 
#  misclassified examples are in total
def steady_elite(elite):
    """ complete this function with an effective method to make the accuracy of 
        the elite stable against what we have, so the errors in the train set """
    """ this is something like a booster where the nets in the elite deal with 
        the most difficult part of the dataset in a distributed way """
    """ 1. test all the elements in the elite against the data misclassified by
        all the elite memebers, with a higher learning rate 
        2. put the new elements in the elite, this means that at the next step 
        we could have a bad elite and in this way some of them will be changed 
        with elements of the population ;)
    """    
    pass;
    
# utility that orders the population of <net, fitness> in descending/ascendig 
#   order wrt the fitness itself
# takes as input:
#        population, the population as tuples of <net, fitness>
#        desc, boolean (that is considered False if not given) that means that 
#           the resulting population is in descending order, ascending otherwise
def sort_by_fitness(population, desc=False):
    return sorted(population, key = itemgetter(1), reverse=False);