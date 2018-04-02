# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:20:25 2018

@author: Emanuele

Test of genetic algorithms: lossless encoding of the nets

Pseudocode:
0. create a popultion of random nets
1. lossless encoding of each net
2. mutation, crossover
3. lossless decoding
4. train, validation, fitness evaluation (, elitism)
4. if conditions are not met, goto 1.
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");

import netgenerator as ngen
import pprint as pp
from utils import encoding as enc

if __name__ == "__main__":
    
    # initial parameters
    i_size = 64;
    o_size = 10;
    pop_size = 20;
    
    # create the population
    population = ngen.rand_population(i_size, o_size, pop_size);
    
    # encode the population as strings (lossless encoding)
    pop_enc = enc.lossless_encoding(population);
    
    print("\n\n");
    print("Initial population:");
    pp.pprint(pop_enc);
    
    # parameters for the evolution's phase
    epochs = 10;
    desired_fitness = .9;
    p_mutation = .85;
    p_crossover = .05;
    selection = "rank";
    mutation = "random-string";
    crossover = "one-point-string";
    
    for _ in range(epochs):
            
        # we maintain the initial number of elements in the population, which is
        #  the classical approach to GA
        for n in range(int(len(population)/2)):
        
            # select two nets and apply crossover
            id1, id2 = ngen.dict_selection[selection](pop_enc);
            p1, p2 = ngen.dict_crossover[crossover](pop_enc[id1], pop_enc[id2], p_crossover); 
            
            # apply mutation
            ngen.dict_mutations[mutation](p1, p_mutation);
            ngen.dict_mutations[mutation](p2, p_mutation);
            
            # substitute the elements in the population
            pop_enc[id1] = p1;
            pop_enc[id2] = p2;
        
        population = enc.lossless_decoding(list(pop_enc.values()));
        pop_enc = enc.lossless_encoding(list(population.values()));
    
    print("\n\n"); 
    print("Final population after", str(epochs), "epochs:")    
    pp.pprint(pop_enc)
        
    
    
    