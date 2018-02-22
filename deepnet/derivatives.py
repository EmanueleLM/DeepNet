# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:16:55 2017

@author: Emanuele

This module calculates the various derivatives of well know functions, and is
used to perform backpropagation in the scenario where we do not know:
    how many activations functions (i.e. layers) we have in our net
    how they are ordered
    
In this module we calculate the first part of each derivative (something like the graphical caluclation)
For example if we want to derive sigma(z) = 1/(1+(exp(-z))), we obtain sigma(z)(1-sigma(z))*d(z), but we skip 
    d(z) since it will be calculated by the next layer
"""

import numpy as np

# derivative of y wrt the exit variable for the L2 loss function
# since we use the function (t-y)^2, we obtain 2(t-y) as derivative
# please note that this one holds it returns a numpy array of size (2,1) since we have two different derivatives for each part
#   (real, imaginary) of the loss, and each of them must be multiplied to different factors according the the chain rule 
def dy_L2(y, t):
    if y.dtype == 'complex' or y.dtype == 'complex64':
        dL_dYr = 2*(np.real(y)-np.real(t)); # dLoss/dYr
        dL_dYi = 2*(np.imag(y)-np.imag(t)); # dLoss/dYi
        return np.array([dL_dYr, dL_dYi]).reshape(2,1); 
    else:
        return 2*np.add(y, -t);

# derivative of y wrt the exit variable for the L1 loss function
# which is basically the signum of the difference between t and y (that's why in regression a lot of weights are zero)
def dy_L1(y, t):
    return -np.sign(t-y);
    
# derivative of y wrt the exit variable for the Cross Entropy loss function  
def dy_cross_entropy(y, t):
    y[y==1] = 1-1e-10;
    y[y==0] = 1e-10;
    return (y-t)/(y*(1-y));

# (partial) derivative of the Kullback-Leibler "measure"
# y is the value predicted by our algorithm
# t is the real value (we are in a supervised scenario)
# we expect that the number of dimensions is the first dimension of both the input (so y.shape[1]=t.shape[1]=dimensions)
# please note that KL can be written as entropy plus cross entropy of target/prediction
def dy_KL(y, t):
    y[y==1] = 1-1e-10;
    y[y==0] = 1e-10;
    entropy = 1+np.log2(y);
    cross_entropy = (y-t)/(y*(1-y));
    return entropy + cross_entropy;

# (partial) derivative of sigmoid function wrt variable z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dsigmoid(z):
    #return (np.exp(-z))/np.power(1+np.exp(-z), 2); # use this version with the modified line in deepnet.py, commented at line (about) 186
    return np.multiply(z,(1-z));

# (partial) derivative of ReLu function wrt variable z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def drelu(z):
    z[z<=0] = 0.;
    z[z>0] = 1.;
    return z;

# (partial) derivative of LeakyReLu function wrt variable z, where \epsilon is set to 0.01
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dleaky_relu(z):
    z[z<=0] = 0.1;
    z[z>0] = 1.;
    return z;

# (partial) derivative of tanh function wrt variable z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dtanh(z):
    return np.multiply((1-z),(1+z));

# (partial) derivative of an exponential fucntion wrt variable z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dexp(z):
    return z;

# (partial) derivative of an linear fucntion wrt variables z, W, b
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dlinear(z):
    return np.ones(z.shape);
