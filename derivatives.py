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

# derivative of Y wrt the exit variable for the L2 loss function
# since we use the function (T-Y)^2, we obtain 2(T-Y) as derivative
# please note that this one holds it returns a numpy array of size (2,1) since we have two different derivatives for each part
#   (real, imaginary) of the loss, and each of them must be multiplied to different factors according the the chain rule 
def dYL2(Y, T):
    if Y.dtype == 'complex' or Y.dtype == 'complex64':
        dL_dYr = 2*(np.real(Y)-np.real(T)); # dLoss/dYr
        dL_dYi = 2*(np.imag(Y)-np.imag(T)); # dLoss/dYi
        return np.array([dL_dYr, dL_dYi]).reshape(2,1); 
    else:
        return 2*np.add(Y, -T);

# derivative of Y wrt the exit variable for the L1 loss function
# which is basically the signum of the difference between T and Y (that's why in regression a lot of weights are zero)
def dYL1(Y, T):
    return -np.sign(T-Y);
    
# derivative of Y wrt the exit variable for the Cross Entropy loss function  
def dYCrossEntropy(Y, T):
    Y[Y==1] = 0.9999;
    Y[Y==0] = 1e-4;
    return (Y-T)/(Y*(1-Y));

# (partial) derivative of sigmoid function wrt variable Z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dSigmoid(Z):
    #return (np.exp(-Z))/np.power(1+np.exp(-Z), 2); # use this version with the modified line in deepnet.py, commented at line (about) 186
    return np.multiply(Z,(1-Z));

# (partial) derivative of ReLu function wrt variable Z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dRelu(Z):
    Z[Z<=0] = 0.;
    Z[Z>0] = 1.;
    return Z;

# (partial) derivative of LeakyReLu function wrt variable Z, where \epsilon is set to 0.01
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dLeakyRelu(Z):
    Z[Z<=0] = 0.1;
    Z[Z>0] = 1.;
    return Z;

# (partial) derivative of tanh function wrt variable Z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dTanh(Z):
    return np.multiply((1-Z),(1+Z));

# (partial) derivative of an exponential fucntion wrt variable Z
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dExp(Z):
    return Z;

# (partial) derivative of an linear fucntion wrt variables Z, W, b
# the partial is not in the common math sense, but in the sense that the "residual" dZ is not calculated at this step
def dLinear(Z):
    return np.ones(Z.shape);