# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:59:58 2018

@author: Emanuele

Looking for a solution that does not involve an explicit research in the hp space
 for the problem presented in [Schmidhuber, Jürgen. "Discovering neural nets with
 low Kolmogorov complexity and high generalization capability." Neural Networks 
 10.5 (1997): 857-873]
 
The problem setting is the following: x \in X is a {0, 1}^{100} vector with 3 
 bits set to 1 (i.e. |X| = (100  3) which is about 160K), while Y is the sum of
 set-in bits (i.e. y=3 \forall x). We assume real weights while in the paper it
 is assumed to be integres from -10K to +10K. Let's see what happens.
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");
import deepnet as dn
import numpy as np

if __name__ == "__main__":
    
    # number of train samples
    sample_size = 3;
    # input/output size
    i_size = 100;
    o_size = 1;
    
    X = np.zeros((sample_size, 100, 1));
    # turn on 3 bits on each sample, randomly
    for x in X:
        for i in np.random.choice(i_size, sample_size):
            x[i] = 1;
            
    # target
    Y = 3*np.ones(X.shape[:1]).reshape(sample_size, 1, 1);
            
    # create the perceptron
    net = dn.DeepNet(i_size, np.array([[1, "linear"]]), "L2");
    net.learning_rate= 3.75e-1;
    
    # initialize the parameters
    for i in range(len(net.weights)):
        net.weights[i] = dn.weights_dict['lecun'](net.weights[i]);
        net.bias[i] = dn.weights_dict['lecun'](net.bias[i]);
    
    # train
    for i in range(sample_size):
        net.batch_backpropagation(X[i], Y[i], batch_size=1);
   
    test_size = 10;
    X_test = np.zeros((test_size, 100, 1));
     
    # test on 10 samples
    # turn on 3 bits on each sample, randomly
    for x in X_test:
        for i in np.random.choice(i_size, sample_size):
            x[i] = 1;
    
    for x in X_test:
        y_hat = net.net_activation(x);
        print("Output expected is 3: output given is ", y_hat);