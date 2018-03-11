# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:20:58 2018

@author: Emanuele
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");

import matplotlib.pyplot as plt
import numpy as np
import utils.utils_digit_recognition as drec
import deepnet as dn

if __name__ == "__main__":
    
#    # initialize the net 
    net = dn.DeepNet(64, np.array([[35, "sigmoid"], [10, "sigmoid"]]), "CrossEntropy");
#    net = dn.DeepNet(64, np.array([[35, "sigmoid"]]), "CrossEntropy");
#    net.add_block(10, "sigmoid", fully_connected=False, connection_percentage=.5);
    net.learning_rate = 2e-3; # set the learning rate        
    
    # initialize train, test, validation
    train_percentage = 70; # percentage of the dataset used for training
    validation_percentage = 50; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)
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
    #X = drec.normalize_data(X);
    X_test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).T;
    Y_test = test_Y;
    X_validation = validation.reshape(validation.shape[0], validation.shape[1]*validation.shape[2]).T;
    Y_validation = validation_Y;
    validation_error = 1; # validation stop metric, initially the error is everywhere
    validation_size = X_validation.shape[1];
      
    # initialize the weights
    for i in range(len(net.weights)): 
        net.weights[i] = dn.weights_dict['flashcard'](net.weights[i], net.weights[0].shape[0], net.weights[-1].shape[1]); 
        net.bias[i] = dn.weights_dict['flashcard'](net.bias[i], net.weights[0].shape[0], net.weights[-1].shape[1]);
          
    # start training   
    epochs = 250;
    for e in range(epochs):
        print("\nEpoch ", e);
        
        # shuffle the train
        X, Y = drec.unison_shuffled_copies(X.T, Y);  
        X = X.T;
        
        # train
        batch_size = 1;
        for n in range(0, X.shape[1] - X.shape[1]%batch_size, batch_size):
            net.batch_backpropagation(X[:,n:n+batch_size].reshape(64, batch_size), Y[n:n+batch_size].reshape(10, batch_size), batch_size);
            
        # validate the results
        number_of_errors_validation = 0;
        for n in range(X_validation.shape[1]):
            if np.argmax(net.net_activation(X_validation[:,n].reshape(64,1))) != np.argmax(Y_validation[n].reshape(10,1)):
                number_of_errors_validation += 1;                
        validation_error = number_of_errors_validation/validation_size;
        print("validation error: ", validation_error);
    
    # test error    
    number_of_errors = 0; # total number of errors on the test set
    test_size = X_test.shape[1];
    for n in range(X_test.shape[1]):
        if np.argmax(net.net_activation(X_test[:,n].reshape(64,1))) != np.argmax(Y_test[n].reshape(10,1)):
            number_of_errors += 1;
    print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");