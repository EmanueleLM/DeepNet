# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:28:20 2018

@author: Emanuele

Parallelization test module: put this module in the main directory of deepnet and run it (it's to test it)
    - soon will be possible to use deepnet and this module as a package
"""

import deepnet as dn
import parallelization.dispatcher as disp
import parallelization.dprediction as dpred
import numpy as np
    
if __name__ == '__main__':
    # without this instruction it does not work ok the IPyhton cosole
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"; 
    # create the nets
    net1 = dn.DeepNet(2, np.array([[3, "linear"], [2, "exp"]]), "L2", verbose=True); 
    net2 = dn.DeepNet(2, np.array([[3, "linear"], [2, "exp"]]), "L2", verbose=True); 
    net3 = dn.DeepNet(2, np.array([[3, "linear"], [2, "exp"]]), "L2", verbose=True); 
    net4 = dn.DeepNet(2, np.array([[3, "linear"], [2, "exp"]]), "L2", verbose=True); 
    # put the nets in a list
    nets = list([net1, net2, net3, net4]);
    for net in nets:
        net.learning_rate = 1e-1;
        for i in range(len(net.weights)): # initialize the weights
            net.weights[i] = dn.weights_dict['lecun'](net.weights[i]); 
            net.bias[i] = dn.weights_dict['lecun'](net.bias[i]);
    print(net1.weights[0]);
    # create the data, samples and labels
    X = np.random.rand(2, 100);
    T = (X**2); 
    # create the pool of processes
    pool, args = disp.processes_dispatchment(nets, X, T, shuffle=False);
    # spawn the processes and return the result of the training
    result = disp.processes_start(pool, args);
    # kill the pool of processes
    disp.processes_end(pool);
    # let's predict the value of a sample
    prediction = dpred.regression(X[:,-1].reshape(net1.weights[0].shape[0], 1), result[0], weighted=False);  
    # show the real prediction againts the one that comes from the pool of nets
    real_prediction = T[:,-1];
    print(net1.weights[0]);
