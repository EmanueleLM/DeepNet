# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:02:37 2018

@author: Emanuele

Information flow thorugh the learning phase in the Iris dataset
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");

import deepnet as dn
import numpy as np
import utils.utils_digit_recognition as drec

from sklearn import datasets

if __name__ == "__main__":
    
        # create the net
        net = dn.DeepNet(4, np.array([[21, "relu"], [3, "sigmoid"]]), "CrossEntropy", verbose=True); 
        net.learning_rate = 1e-4; # set the learning rate 
        
        
        # import data
        iris = datasets.load_iris();
        X = iris.data;
        y = drec.binarization(iris.target)[:,:3];
        
        # split train and test (70% train, 30% test)
        train, test = drec.data_split(X, 70);
        train_Y, test_Y = drec.data_split(y, 30);
               
        # initialize the weights
        for i in range(len(net.weights)): 
            net.weights[i] = dn.weights_dict['lecun'](net.weights[i]); 
            net.bias[i] = dn.weights_dict['lecun'](net.bias[i]);
            
        # train (30 epochs)
        for e in range(30):
            
            for n in range(X.shape[0]):                
                net.backpropagation(X[n].reshape(4,1), y[n].reshape(3,1));
                
        # test the model on unseen data
        test_errors = 0;
        test_size = test.shape[0];
        
        for n in range(test.shape[1]):
            if np.argmax(net.net_activation(test[n].reshape(4,1))) != np.argmax(test_Y[n].reshape(3,1)):
                test_errors += 1;
                
        print("The error percentage is ", test_errors/test_size, ": ", test_errors,
              " errors out of ", test_size, " samples on test set.");
