# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:03:32 2018

@author: Emanuele

MNIST test script
"""

# solve the relative import ussues
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../deepnet/");
import deepnet as dn
import utils.utils_digit_recognition as drec
import numpy as np

if __name__ == "__main__":
    
    # create the net and use a 50% connectivity (pre defined dropout)
    net = dn.DeepNet(input_size= 784, 
                     layers=np.array([[35, "relu"], [10, "sigmoid"]]), 
                     loss="CrossEntropy",
                     verbose=True); 
    
    net.learning_rate = 1e-4;   
    
    # initialize train, test, validation
    train_percentage = 90;
    train_digits = np.loadtxt('data/mnist/mnist_train.csv', 
                              delimiter=',', 
                              skiprows=40000);
                              
    test_digits = np.loadtxt('data/mnist/mnist_test.csv', 
                             delimiter=',', 
                             skiprows=0); 
                             
    train_size = len(train_digits); 
    
    # shuffle together inputs and supervised outputs
    images, targets = drec.unison_shuffled_copies(train_digits[:,1:], train_digits[:,0].astype(int)); 
    train, validation = drec.data_split(images, train_percentage);
    train_Y, validation_Y = drec.data_split(targets, train_percentage); 
    
    # binarize the prediction    
    test, test_Y = drec.unison_shuffled_copies(test_digits[:,1:], test_digits[:,0].astype(int));
    train_Y = drec.binarization(train_Y);
    test_Y = drec.binarization(test_Y); 
    validation_Y = drec.binarization(validation_Y); 
    X = train.reshape(train.shape[0], train.shape[1]).T;
    Y = train_Y;
    
    X_test = test.reshape(test.shape[0], test.shape[1]).T;
    Y_test = test_Y;
    X_validation = validation.reshape(validation.shape[0], validation.shape[1]).T;
    Y_validation = validation_Y;

    epochs = 30;
    
    # validation stop metric, initially the error is everywhere
    validation_error = 1; 
    validation_size = X_validation.shape[1];
     
    # initialize the weights    
    for i in range(len(net.weights)):
        net.weights[i] = dn.weights_dict['flashcard'](net.weights[i], net.weights[0].shape[0], net.weights[-1].shape[1]); 
        net.bias[i] = dn.weights_dict['flashcard'](net.bias[i], net.weights[0].shape[0], net.weights[-1].shape[1]);
    
    # train
    num_layers = len(net.weights);
    for e in range(epochs):
        print("\nEpoch ", e);
        
        batch_size=10;              
        for n in range(0, X.shape[1]-X.shape[1]%batch_size, batch_size):
            
#            # rudimental dropout technique
#            net.mask = list();
#            for w in net.weights:
#                net.mask.append(np.random.choice([0.,1.], size=w.shape, p=[1.-.5, .5]));
            
            net.batch_backpropagation(X[:,n:n+batch_size].reshape(784, batch_size), 
                                      Y[n:n+batch_size].reshape(10, batch_size), 
                                      batch_size);
                                                 
        number_of_errors_validation = 0;
            
        for n in range(X_validation.shape[1]):
            
            if np.argmax(net.net_activation(X_validation[:,n].reshape(784,1))) != np.argmax(Y_validation[n].reshape(10,1)):
                
                number_of_errors_validation += 1;
                
        validation_error = number_of_errors_validation/validation_size;
        print("validation error: ", validation_error);

    # test the accuracy on unseen data           
    number_of_errors = 0; 
    test_size = X_test.shape[1];
    
    for n in range(X_test.shape[1]):
        
        if np.argmax(net.net_activation(X_test[:,n].reshape(784,1))) != np.argmax(Y_test[n].reshape(10,1)):
            
            number_of_errors += 1;
            
    # accuracy on test set
    print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");     
