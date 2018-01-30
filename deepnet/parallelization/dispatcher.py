# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:34:16 2018

@author: Emanuele

This module uses different modules to train in parallel several neural networks;
    - the number of epochs each net is trained can be much less than the number you would use for a single net (parallelization). 

The initial dataset is split in chunks and feeded to each network randomly by the dispatcher (this increases generalization)
    - can't mix the datasets since sgd is deterministic we would obtain a rig of identical nets.   
"""

import multiprocessing
import numpy as np
import parallelization.dtrain as dt
import psutil # used to count the number of logical CPUs

# create several neural networks and dispatch their training on different processes
#   it is not already possible to dispatch them on a different device such as gpus etc.: I'm working on this ;)
# takes as input:
#   nets, a list of neural networks tranied on different samples (at least one)
#   X, the entire train set imported as (dimensions, samples) <- standard in neural networks (each column a sample)
#   T, the entire train label set , imported as (dimensions, samples) <- standard in neural networks (each column a sample)
#   shuffle, a boolean that eventually shuffle the dataset before dispatching it to the different nets (set to False, i.e. it doesn't shuffle X,T)
#   CPUs_percentage, the number of processes spawned in percentage wrt the number of logical CPUs available to the machine (set to 1., i.e. all of them)
# returns:
#   the pool of processes we will spwan and the arguments arranges in a 'iterable' that can be passed to a Pool.starmap() function
def processes_dispatchment(nets, X, T, shuffle=False, CPUs_percentage=1.):
    if shuffle is True:
        # shuffle the data
        assert X.shape[1] == T.shape[1]; # same number of elements
        p = np.random.permutation(X.shape[1]);
        X, T = X[:,p], T[:,p];
    num_nets = len(nets);
    chunk_dim = int(X.shape[1]/num_nets); # dimension of each chink of train data for each net
    # assign each chunk to a position in a list, used during the map procedure
    X_n = list([X[:,i:i+chunk_dim] for i in range(0, num_nets*chunk_dim, chunk_dim)]);
    T_n = list([T[:,i:i+chunk_dim] for i in range(0, num_nets*chunk_dim, chunk_dim)]);
    number_cpus = np.max([1, np.min([int(CPUs_percentage*psutil.cpu_count()), num_nets])]); # max: at least one cpu, min: at most a process for each net
    args = list();
    for n in range(num_nets):
        args.append((nets[n], X_n[n], T_n[n]));
    return multiprocessing.Pool(number_cpus), args;

# start a pool of processes by passing the input arguments
# takes as input:
#   pool, the pool of porcesses, whose type is multiprocessing.Pool
#   args, which is an iterable that contains the input of the functions we want to execute
# returns:
#   the results of the parallelization process and the pool, if we want to eventually kill the processes manually (or for any other reason)
def processes_start(pool, args):
    # collect the results of the training
    result = pool.starmap(dt.train, args);
    print("Training has finished");
    return result, pool;

# kill the pool of processes
# takes as input:
#   pool the pool of processes, type multiprocessing.Pool
def processes_end(pool):
    pool.close();
    pool.join();
