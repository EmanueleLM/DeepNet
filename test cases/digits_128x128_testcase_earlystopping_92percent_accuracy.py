# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:55:31 2017

@author:  Emanuele

Train/Test/Validation with early stoping technique

We use 20k out of 40k samples, as many epochs as we want (100 as upper bound) and validation to stop the learning
We use a 2 layer net with 500 neurons, both sigmoid in the hidden and in the output layer, a learning rate of 0.008
a momentum of 0.006 (tune 'em as you want)
We obtain an accuracy of about 92% (np dropout, and probably a good momentum woul help a lot)
If we use just 2000 samples, with the same parameters we get something like 84% of accuracy on unseen data

Use this code just below the code on your deepnet.py (it should be done in another file indipendently importing
the module, I don't have both the time and the will)
"""

import utils.utils_digit_recognition as drec

train_percentage = 60; # percentage of the dataset used for training
validation_percentage = 20; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)

digits = np.loadtxt('train.csv', delimiter=',', skiprows=40000); # import the dataset

train_size = len(digits); # train size is the number of samples in the digits' dataset

images, targets = drec.unison_shuffled_copies(digits[:,1:], digits[:,0].astype(int)); # shuffle together inputs and supervised outputs

train, test = drec.data_split(images, train_percentage);# split train adn test
train_Y, test_Y = drec.data_split(targets, train_percentage); # split train and test labels

validation, test = drec.data_split(test, validation_percentage);
validation_Y, test_Y = drec.data_split(test_Y, validation_percentage);

train_Y = drec.binarization(train_Y); # binarize both the train and test labels
test_Y = drec.binarization(test_Y); # ..
validation_Y = drec.binarization(validation_Y); # ..


X = train.reshape(train.shape[0], train.shape[1]).T;
Y = train_Y;

#X = drec.normalizeData(X);
#Y = drec.normalizeData(Y.T).T;

X_test = test.reshape(test.shape[0], test.shape[1]).T;
Y_test = test_Y;

X_validation = validation.reshape(validation.shape[0], validation.shape[1]).T;
Y_validation = validation_Y;


#X_test = drec.normalizeData(X_test);
#Y_test = drec.normalizeData(Y_test.T).T;

""" Train with full batch (size of the batch equals to the size of the dataset) """
epochs = 100;
validation_error = 1; # validation stop metric, initially the error is everywhere
validation_size = X_validation.shape[1];
for e in range(epochs):
    print((epochs-e)," epochs left");
    for n in range(X.shape[1]):
        net.backpropagation(X[:,n].reshape(784,1), Y[n].reshape(10,1));
        number_of_errors_validation = 0;
    for n in range(X_validation.shape[1]):
        if np.argmax(net.net_activation(X_validation[:,n].reshape(784,1))) != np.argmax(Y_validation[n].reshape(10,1)):
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
    if np.argmax(net.net_activation(X_test[:,n].reshape(784,1))) != np.argmax(Y_test[n].reshape(10,1)):
        number_of_errors += 1;
print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");