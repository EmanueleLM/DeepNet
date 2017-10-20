# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:27:59 2017

@author: Emanuele

Test case for a 2 layers feed forward network:
    in the first layer we have the input which is a two dimensional input {x1, x2}
    then we have a weights matrix whose dimension is (2,2)
    the first neurons' layer has two sigmoids as activation functions
    then we have another weights' matrix whose size is (2,1) and another sigmoid as output
    
    with the initialization you find in the script, we obtained the following weights/biases updates
    dW1 = array([[ 0.00864725,  0.0012725 ],[ 0.0172945 ,  0.00254501]]);
    dW2 = array([[ 0.18233193],[ 0.19012863]]);
    dB1 = array([[ 0.00864725],[ 0.0012725 ]]);
    dB2 = array([[ 0.1914097]]);
    
    the learning rate was set to 1
    
    all the calculations for the results have been made also by hand, so they should be correct ;)
    
    Put this code at the end of the deepnet.py module and execute it
"""
net = DeepNet(2, np.array([[2, "sigmoid"], [1, "sigmoid"]]), "L2"); # create a net with this simple syntax
net.learning_rate = 1; # set the learning rate to 1

X = np.array([[[1.],[2.]]]);
T = np.array([[0.]]);
net.W[0] = np.array([[1.,1.],[1.,2.]]);
net.W[1] = np.array([[1.],[1.]]);
net.Bias[0] = np.array([[0.],[0.]]);
net.Bias[1] = np.array([[0.]]);

print("Initial weights and biases");
print(net.W);
print(net.Bias);

dW, dB = net.backpropagation(X[0], T);

print("Updated weights and their update");
print(net.W, dW);
print("Updated biases and their update")
print(net.Bias, dB);