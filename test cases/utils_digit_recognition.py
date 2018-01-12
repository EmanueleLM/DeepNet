# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:13:31 2017

@author: Emanuele
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits # import the digit dataset, a dataset of digits in a greyscale matrix each, 8x8

# function that shuffles together two arrays (whose len is equal and it's the first element of the array in np.shape)
# takes as input 
#   a,b the two arrays
# returns
#   their shuffled versions
def unison_shuffled_copies(a, b):
    assert len(a) == len(b);
    p = np.random.permutation(len(a));
    return a[p], b[p];

# function that plots a digit starting from a 8*8 matrix of greyscales
# takes as input
#   m, the matrix that contains the greyscale of the digit
def plot_digit(m):
    plt.gray();
    plt.matshow(m); 
    plt.show();
    
# function that split train and test from a dataset, according to a given percentage
# takes as input
#   data, the dataset we want to split between train/test
#   train_percentage, the percentage of split train/test
# returns
# thetrain adn test datasets, splitted accoridngly
def data_split(data, train_percentage):
    last_index_train = int(len(data)*(train_percentage/100)); # calculate the number of train samples wrt the number of data samples
    return data[:last_index_train], data[last_index_train:];

# function that transorms the labels'dataset into a matrix of binary vector
# i.e. it binarize each labels
    # '0' is turned into [1,0,0,0,0,0,0,0,0,0]
    # '1' is turned into [0,1,0,0,0,0,0,0,0,0]
    # '2' is turned into [0,0,1,0,0,0,0,0,0,0]
    # '3' is turned into [0,0,0,1,0,0,0,0,0,0]
    # '4' is turned into [0,0,0,0,1,0,0,0,0,0]
    # '5' is turned into [0,0,0,0,0,1,0,0,0,0]
    # '6' is turned into [0,0,0,0,0,0,1,0,0,0]
    # '7' is turned into [0,0,0,0,0,0,0,1,0,0]
    # '8' is turned into [0,0,0,0,0,0,0,0,1,0]
    # '9' is turned into [0,0,0,0,0,0,0,0,0,1]
# returns
#   the binarized dataset
def binarization(Y):
    res = np.array([[0 for i in range(10)] for j in range(len(Y))]); # 10 possible labels means vectors of size 10 each
    for n in range(len(Y)):
        res[n][Y[n]] = 1;
    return res;

def vectorize(X):
    return np.array(X.reshape(X.shape[0], X.shape[1]*X.shape[2]));

# function that normalize the data
# takes as input
#   X, the input matrix, whose shape is (m, n, 1)
# returns
# normalized X, with the same shape
def normalize_data(X):
    #X = X.reshape(X.shape[0], X.shape[1], 1);
    for i in range(X.shape[0]):
        X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i]); # normalize both input and output
    return X;