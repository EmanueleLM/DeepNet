# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:48:29 2017

@author: Emanuele


Train/Test/Validation with early stoping technique
We obtain, on the digit dataset, an accuracy of about 96% on unseen data

Use this code just below the code on your deepnet.py (it should be done in another file indipendently importing
the module, I don't have both the time and the will)

In order to obtain those results, we set learning_rate = 0.02, no momentum
"""

net = DeepNet(64, np.array([[128, "sigmoid"], [10, "sigmoid"]]), "L2"); # create the net
for i in range(len(net.weights)): #initialize the weights
    net.weights = weights_dict['lecun'](net.weights[i]); 
    net.bias = weights_dict['lecun'](net.bias[i]);

import utils_digit_recognition as drec

train_percentage = 60; # percentage of the dataset used for training
validation_percentage = 20; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)

digits = drec.load_digits(); # import the dataset

train_size = len(digits.images); # train size is the number of samples in the digits' dataset

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
#Y = drec.normalize_data(Y.T).T;

X_test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).T;
Y_test = test_Y;

X_validation = validation.reshape(validation.shape[0], validation.shape[1]*validation.shape[2]).T;
Y_validation = validation_Y;


#X_test = drec.normalize_data(X_test);
#Y_test = drec.normalize_data(Y_test.T).T;

""" Train with full batch (size of the batch equals to the size of the dataset) """
epochs = 100;
validation_error = 1; # validation stop metric, initially the error is everywhere
validation_size = X_validation.shape[1];
for e in range(epochs):
    print((epochs-e)," epochs left");
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
        print("validation error: ", validation_error);

""" test how much we are precise in our prediction """
number_of_errors = 0; # total number of errors on the test set
test_size = X_test.shape[1];
for n in range(X_test.shape[1]):
    if np.argmax(net.net_activation(X_test[:,n].reshape(64,1))) != np.argmax(Y_test[n].reshape(10,1)):
        number_of_errors += 1;
print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");
