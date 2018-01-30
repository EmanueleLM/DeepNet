# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:55:31 2017

@author:  Emanuele

Train/Test/Validation with early stoping technique

This code is meant to give the opportunity to test your net against a letter recognition problem:
 you have each input which is of shape (17,1) where the first one is the label (from 'a' to 'z')
 while the other 16 dimensions are complex feature (not in the math sense :/ ).

Use this code just below the code on your weinernet.py (it should be done in another file indipendently importing
the module, I don't have both the time and the will)
""" 

import utils_digit_recognition as drec
import numpy as np

train_percentage = 60; # percentage of the dataset used for training
validation_percentage = 20; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)
digits = np.loadtxt('letters_rec/lr.data', dtype='str'); # import the dataset, please use skiprows=n if you want to avoid using 20K samples
digits = np.array([d.split(',') for d in list(digits)]);
for d in digits:
    d[0] = ord(d[0])-65; # convert the ascii output to a number from 0 ('a') to 25 ('z') 
train_size = len(digits); # train size is the number of samples in the digits' dataset
images, targets = drec.unison_shuffled_copies(digits[:,1:].astype(int), digits[:,0].astype(int)); # shuffle together inputs and supervised outputs
train, test = drec.data_split(images, train_percentage);# split train adn test
train_Y, test_Y = drec.data_split(targets, train_percentage); # split train and test labels
validation, test = drec.data_split(test, validation_percentage);
validation_Y, test_Y = drec.data_split(test_Y, validation_percentage);
train_Y = drec.binarization(train_Y); # binarize both the train and test labels
test_Y = drec.binarization(test_Y); # ..
validation_Y = drec.binarization(validation_Y); # ..
X = train.reshape(train.shape[0], train.shape[1]).T;
Y = train_Y;
X_test = test.reshape(test.shape[0], test.shape[1]).T;
Y_test = test_Y;
X_validation = validation.reshape(validation.shape[0], validation.shape[1]).T;
Y_validation = validation_Y;

""" Train with full batch (size of the batch equals to the size of the dataset) """
epochs = 20;
validation_error = 1; # validation stop metric, initially the error is everywhere
validation_size = X_validation.shape[1];
for e in range(epochs):
    print((epochs-e)," epochs left");
    for n in range(X.shape[1]):
        net.backpropagation(X[:,n].reshape(16,1), Y[n].reshape(26,1));
        number_of_errors_validation = 0;
    for n in range(X_validation.shape[1]):
        if np.argmax(net.net_activation(X_validation[:,n].reshape(16,1))) != np.argmax(Y_validation[n].reshape(26,1)):
            number_of_errors_validation += 1;
    if float(number_of_errors_validation/validation_size) > validation_error:
        break;
    else:
        validation_error = number_of_errors_validation/validation_size;
        print("validation error: ", validation_error);

""" test how much we are precise in our prediction """
number_of_errors = 0; # total number of errors on the test set
test_size = X_test.shape[1];
for n in range(X_test.shape[1]):
    if np.argmax(net.net_activation(X_test[:,n].reshape(16,1))) != np.argmax(Y_test[n].reshape(26,1)):
        number_of_errors += 1;
print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");
