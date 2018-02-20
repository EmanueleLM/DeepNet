# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:37:21 2017

@author: Emanuele

This module introduces one of the main features of the DeepNet core: the exploration of the space
    of the possible nets wrt a fixed problem with a genetic algorithm. We start with a population of 
    n cromosomes that are n neural nets with different topologies and we combine them with specific
    genetic operators in order to find the best suited for the problem
Please note that this code is not natively parallelized, but we aim at developing a module that can do
    training of the various nets in parallel. 
    The aim of this project (to me) is not to learn parallel computation from scratch, 
    but how to use genetic algorithm to generate good nets
"""

import numpy as np
import deepnet as dn
import genetic.crossover as cr
import genetic.mutation as mu
import genetic.selection as sel
import genetic.elitism as el

MAX_NUM_LAYERS = 4; # maximum number of layers that a generic net generated by this code can have (we use C like code for 'global variables')
MIN_NUM_LAYERS = 2; # maximum number of layers that a generic net generated by this code can have
MAX_NUM_NEURONS = 10; # maximum number of neurons a layer can have
MIN_NUM_NEURONS = 5; # minimum number of neurons a layer can have

dict_crossover = {"one-point": cr.one_point_crossover}; # dict for the various crossover methods
dict_mutations = {"random": mu.random_mutation}; # dict for the various mutations methods
dict_selection = {"rank": sel.rank_selection, "roulette": sel.roulette_selection}; # dict for the various selection methods

# this function is used to generate a population of n neural networks
# you have just to specify the input size, the output size and how many neural net we want to generate
# takes as input:
#   input_size, the number of inputs we have (i.e. the dimensions of the image of the function we want to optimize)
#   output_size, the number of outputs of the function we want to optimize
#   population_size, the number of neural net we want to use to find the best one
#   fully_connect, a boolean that is true iff we want that the network space we explore considers just fully connected topologies
#   connection_perc, how many (in percentage) connections are not cut off during the creation of the net, in a non fully connected net (this one is considere if fully_connected is True)
# returns:
#   net_population, a list that contains the nets of the fresh new population
def random_population(input_size, output_size, population_size, fully_connect=True, connection_perc = .5):
    net_population = list(); # a list since the elements inside each net are eterogeneous
    for n in range(population_size):
        n_layers = np.random.randint(MIN_NUM_LAYERS, MAX_NUM_LAYERS);
        dict_act_size = len(dn.activations_dict); # size of the activations' dictionary
        loss_dict_size = len(dn.loss_dict); # size of the losses' dictionary
        layer = np.array([]);
        for i in range(n_layers-1):            
            layer = np.append(layer, np.array([np.random.randint(MIN_NUM_NEURONS, MAX_NUM_NEURONS), list(dn.activations_dict.keys())[np.random.randint(0, dict_act_size)]]));      
        layer = np.append(layer, np.array([output_size, list(dn.activations_dict.keys())[np.random.randint(0, dict_act_size)]])); # output layer
        net_population.append(dn.DeepNet(input_size, layer.reshape(n_layers, 2), list(dn.loss_dict.keys())[np.random.randint(0, loss_dict_size)], verbose=True, fully_connected = fully_connect, connection_percentage=connection_perc)); # append an element of type DeepNet
    return net_population;

#this function returns, for a population of n nets, the fitness against a given problem
# in terms of accuracy and size of the net
#each net has its fitness evaluated in this way: the size of the net in terms of parameters, 
#    multiplied by a parameter w0 (gives a certain importance to this factor), plus the accuracy
#    times another parameter w1, such that (w0+w1)=1 (i.e. the fitness is a convex combination of the size of the net and the accuracy)
#    Obviously we should set w1>>w0 s.t. a little net with low accuracy cannot be better than a huge net with good accuracy
# return a list [[net], fitness]^+
# takes as input:
#   population, a list that contains all the nets (not the couples <net, fitness>)
#   w0, the 'importance' we assign to the accuracy of our model
#   w1, the importance we assign to the size of our model
# returns:
#   a list of tuples <net, fitness>, where each net has its own fitness calculated
def evaluate_fitness(nets, w0, w1):
    population = list([[n, .0] for n in nets]); # put the couples <net, fitness>, initially fintess = 0 forall nets
    num_parameters_per_net = np.array([n.number_of_parameters() for n in nets]); # extract for each net the number of parameters
    max_num_parameters = np.max(num_parameters_per_net); # extract the size of the biggest net
    for n in range(len(nets)):
        """ in this part we need to evaluate the fintess of a single network,
            this means that we need to test the net against a problem and extract
            the validation error. then we will calculate a measure of the size of the net
            and we use those two parameters in a convex combination to calculate the 
            fitness of each net """
        errors = (validate_net(population[n][0]));
        population[n][1] = w0*errors + w1*(1-(num_parameters_per_net[n]/max_num_parameters)); # calculate the fitness, this part should be massively parallelized
    return population;

# function that validate the net i.e. calculate the validation error for a network
# in this case we use the 8x8 handwritten dataset in scikit-learn
#   we must implement a way to test it against a generic problem
# takes as input:
#   net, the net used to validate
#   misclassified, a boolean ((supposed as False)) that tells whether the function has to return the samples that are misclassified by the net 
# returns:
#   (eventually) the samples that are misclassified by the net or the number of errors (as a fraction of 1)
def validate_net(net, misclassified=False):
    for i in range(len(net.weights)): #initialize the weights
        net.weights[i] = dn.weights_dict['lecun'](net.weights[i]); 
    import utils.utils_digit_recognition as drec
    train_percentage = 60; # percentage of the dataset used for training
    validation_percentage = 20; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)
    digits = drec.load_digits(); # import the dataset
    images, targets = drec.unison_shuffled_copies(digits.images, digits.target); # shuffle together inputs and supervised outputs
    train, test = drec.data_split(images, train_percentage);# split train adn test
    train_Y, test_Y = drec.data_split(targets, train_percentage); # split train and test labels
    validation, test = drec.data_split(test, validation_percentage);
    validation_Y, test_Y = drec.data_split(test_Y, validation_percentage);
    train_Y = drec.binarization(train_Y); # binarize both the train and test labels
    test_Y = drec.binarization(test_Y); # ..
    validation_Y = drec.binarization(validation_Y); # ..
    X = train.reshape(train.shape[0], train.shape[1]*train.shape[2]).T;
    Y = train_Y;
    X = drec.normalize_data(X);
    X_test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).T;
    Y_test = test_Y;
    X_validation = validation.reshape(validation.shape[0], validation.shape[1]*validation.shape[2]).T;
    Y_validation = validation_Y;
    epochs = 10;
    validation_error = 1; # validation stop metric, initially the error is everywhere
    validation_size = X_validation.shape[1];
    for e in range(epochs):
        for n in range(X.shape[1]):
            net.backpropagation(X[:,n].reshape(64,1), Y[n].reshape(10,1));
            number_of_errors_validation = 0;
        for n in range(X_validation.shape[1]):
            if np.argmax(net.net_activation(X_validation[:,n].reshape(64,1))) != np.argmax(Y_validation[n].reshape(10,1)):
                number_of_errors_validation += 1;
        if float(number_of_errors_validation/validation_size) > validation_error:
            break;
        else:
            validation_error = number_of_errors_validation/validation_size;
            #print("validation error: ", validation_error);
    number_of_errors = 0; # total number of errors on the test set
    test_size = X_test.shape[1];
    for n in range(X_test.shape[1]):
        if np.argmax(net.net_activation(X_test[:,n].reshape(64,1))) != np.argmax(Y_test[n].reshape(10,1)):
            number_of_errors += 1;
    #print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");
    return number_of_errors/test_size;

# function that evolves a population according a genetic algorithm
# takes as input:
#   population_size, the number of nets to evolve
#   epochs, number of epochs we perform the genetic algorithm
#   input_size, the input size of each network (i.e. input dimension)
#   output_size, the output of each network (i.e. number of neurons in the last layer)
#   selection_type, a string that indicates the selection method used, taken by dict_selection
#   crossover_type, a string that indicates the crossover method used, taken by crossover_dict
#   mutation_type,  a string that indicates the mutation method used, taken by mutation_dict
#   fully_connected, a boolean (set to False) that indicates whether the net is fully connected or not
#   connection_percentage, the percentage (from 0 to 1) of connections in the averall network
#   elite_size, the number of elements in the elite (set to 3)
#   crossover_probability, the probability that two given nets, selected by a selection method, are subject to crossover
#   mutation_probability, a probability that a net is subjecte to mutation (in each of its part)
# returns:
#   the population evolved with the genetic algorithm
#   the elite from the population
# please note that this function is a 'blob': anyway the parameters for both a population of nets and its evolving strategy are really al lot so use it 
#   iff you want to have a fast evolution method. To do a good evolution of a population, check and rewrite the """Test Part"""
def evolve_population(population_size, epochs, input_size, output_size, selection_type, crossover_type, mutation_type, fully_connected=False, connection_percentage=.5, elite_size=3, crossover_probability=.8, mutation_probability=.05):
    nets = random_population(input_size, output_size, population_size, fully_connect=fully_connected, connection_perc=connection_percentage); # create the population 
    if elite_size > population_size:
        elite_size = population_size;    
    population = evaluate_fitness(nets, 1., .0); # train and validate the initial population
    for e in range(epochs):
        print("exploring epoch", e);
        elite, population = el.elitism(population, elite_size, ordered=False); # select the elite in the population and 'save' it from mutation and crossover
        new_population = list(); # result of the crossover/mutation on the old population
        for n in range(int(population_size/2)):
            id1, id2 = dict_selection[selection_type](population, ordered=True);
            p1, p2 = dict_crossover[crossover_type](population[id1][0], population[id2][0], crossover_probability); # apply crossover 
            dict_mutations[mutation_type](p1, mutation_probability);
            dict_mutations[mutation_type](p2, mutation_probability);
            new_population.append([p1, validate_net(p1)]);
            new_population.append([p2, validate_net(p2)]);
        population = new_population;
        population = evaluate_fitness(list([p[0] for p in population]), 1., .0); # train and validate the initial population
        population[-elite_size:] = elite; # substitute the worst elements with the elite 
    return population, elite;

""" Test part """
verbose = True;
if verbose:
    population_size = 2; # number of elements in our starting population
    epochs = 5; # number of epochs we want to iterate the ga routine
    nets = random_population(64, 10, population_size, fully_connect=False, connection_perc=.5); # create the population 
    crossover_probability = 1.;
    mutation_probability = .9;
    elite_size = 2;    
    population = evaluate_fitness(nets, 1., .0); # train and validate the initial population
    for e in range(epochs):
        print("exploring epoch", e);
        elite, population = el.elitism(population, elite_size, ordered=False); # select the elite in the population and 'save' it from mutation and crossover
        new_population = list(); # result of the crossover/mutation on the old population
        for n in range(int(population_size/2)):
            id1, id2 = dict_selection["rank"](population, ordered=True);
            p1, p2 = dict_crossover["one-point"](population[id1][0], population[id2][0], crossover_probability); # apply crossover 
            """ we disable the mutation since we are not sure it benefits the evolution """
            #dict_mutations["random"](p1, mutation_probability);
            #dict_mutations["random"](p2, mutation_probability);
            new_population.append([p1, 1.]);
            new_population.append([p2, 1.]);
        population = new_population;
        population = evaluate_fitness(list([p[0] for p in population]), 1., .0); # train and validate the initial population
        population[-elite_size:] = elite; # substitute the worst elements with the elite 
        print(elite);