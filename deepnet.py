# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:59:16 2017

@author: Emanuele

Deep network learner with parameter tuning and all sort of optimizations

The starting point is just a feed forward neural net where we add much more layers in depth.
"""

import numpy as np
import activations as act
import loss as ls
import derivatives as de
import weights as we
import data as da

# we use a dictionary to handle each layer's activation function: we will need just to know the info contained in
#   DeepNet.layers to invoke the right function!
# Use this struct in this way: 
#   define a function in activations.py (imported let's say as act), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": act.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)
activations_dict = {"relu": act.relu, "sigmoid": act.sigma, "tanh": act.tanh, "leakyrelu": act.leakyRelu};

# the same method is employed for the choice of the loss function
# Use this struct in this way: 
#   define a function in loss.py (imported let's say as ls), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": ls.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)+
loss_dict = {"L1": ls.lossL1, "L2":ls.lossL2, "CrossEntropy": ls.lossCrossEntropy};

# the same method is employed for the choice of the derivatives function
# Use this struct in this way: 
#   define a function in derivatives.py (imported let's say as de), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": de.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)
derivatives_dict ={"L1": de.dYL1, "L2": de.dYL2, "CrossEntropy": de.dYCrossEntropy, "relu": de.dRelu, "leakyrelu": act.leakyRelu,"sigmoid": de.dSigmoid, "tanh": de.dTanh};

# the same method is employed for the choice of the weights' initialization
# Use this struct in this way: 
#   define a function in weights.py (imported let's say as we), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": we.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)+
weights_dict = {"random": we.randomWeights, "uniform":we.uniformWeights, "lecun": we.lecunWeights, "unitary": we.unitaryWeights};

# =============================================================================
#  class that models a deep network with multiple layers and different acrivation functions
#  take as input (the __init__ function):
#      input_size: size of a generic input sample
#      layers: a numpy.array (not a list since we will use numpy.size on it) that contains a number and a string: the former specifies the number of neurons on that layer
#          the latter specifies the activation function (e.g. a valid layers is list([[1,"relu"][5, "sigmoid"]]))
#      loss: the loss function selected for the backpropagation phase
# =============================================================================
class DeepNet(object):
    def __init__(self, input_size, layers, loss):
        self.W = list(); # list that contains all the weights in the net (we use a list and np.array for each matrix of weights, for efficiency reasons)
        self.W.append(np.array(np.zeros([input_size, np.int(layers[0][0])])));
        self.Bias = np.zeros([layers.shape[0], 1]);
        self.activations = np.array(layers[:,1]);
        self.loss = loss;
        self.learning_rate = 0.08; # default learning rate for each iteration phase
        for l in range(len(layers)-1):
            self.W.append(np.array(np.zeros([np.int(layers[l][0]), np.int(layers[l+1][0])])));
        print("\nNetwork created succesfully!")
        self.explainNet();
        
    # getter and setters for the net elements
    # Setters:
    #   weights
    def setWeights(self, W, layer):
        self.W[layer] = W;
    #   bias
    def setBias(self, B):
        self.Bias = B;
    #   activations    
    def setActivation(self, layer, activation):
        self.activations[layer] = activation;
    #   Loss    
    def setLoss(self, l):
        self.loss = l;
    #   learning rate    
    def setLearningRate(self, a):
        self.learning_rate = a;
    # 
    # Getters:
    #   weights
    def getWeights(self, layer):
        return self.W[layer];
    #   bias
    def getBias(self):
        return self.Bias;
    #   activations    
    def getActivation(self, layer):
        return self.activations[layer];
    #   Loss    
    def getLoss(self):
        return self.loss;
    #   learning rate    
    def getLearningRate(self):
        return self.learning_rate;
            
    # function that explains the net: this means that it describes the network
    # in terms of input, layers and activation functions
    def explainNet(self):
        print("\nThe network has ", self.W[0].shape[0], " input(s) and ", len(self.W), " layers.");
        for l in range(len(self.W)):
            print("The layer number ", l+1, " has ", self.W[l].shape[0], " input(s) per neuron, ", self.W[l].shape[1], " neuron(s) and ", self.activations[l], " as activation function.");
        print("The loss function selected is ", self.loss);
        
    # function that activates a single, given layer and, based on its activation function, returns the desired output
    # takes as input
    #   the input X as a vector
    #   the layer where we want the activation to happen
    # returns
    #   the matrix of the activations (even one element for example in single output nn)
    def activation(self, X, layer):
        Z = np.dot(self.W[layer].T,X)+self.Bias[layer]; # activate (linearly) the input
        return activations_dict[self.activations[layer]](Z); # activate the activation function of each layer using the vocabulary defined at the beginning            
    
    # perform activation truncated to a given layer
    # please note that for a L layers nn, we will have L activations Z(1), .., Z(4)
    #   whose dimension is equal to [m,1], where m is the number of neurons the layer ends with 
    def partialActivation(self, X, layer):
        I = X;
        for l in range(layer+1):
            I = self.activation(I, l);
        return I;
    
    # function that activates sequentially all the layers in the net and returns the desired output (a.k.a. "forward")
    # takes as input
    #   the input X as a vector
    # returns
    #   the output vector (even one element for example in single output nn)    
    def netActivation(self, X):
        res = X;
        for l in range(len(self.W)):
            res = self.activation(res, l);
        return res; 
    
    # function that calculates the loss for a given input, output and a loss function
    # takes as input:
    #   the prediction Y (can be calculated as self.netActivation(input))
    #   the target value 
    # returns:
    #   the loss with the formula specified in self.loss, using the dictionary structure loss_dict to invoke the correct function (see above for more info)
    def calculateLoss(self, Y, T):
        return loss_dict[self.loss](Y, T)
    
    # function that performs a step of back propagation of the weights update from the output
    #   (i.e. the loss error) to the varoius weights of the net
    # we use the chain rule to generalize the concept of derivative of the loss wrt the weights
    def backpropagation(self, X, T):  
        dW = list(); # list of deltas that are used to calculate weights' update
        dB = np.array([]); # array of deltas that are used to calculate biases' update
        y_hat = self.netActivation(X); # prediction of the network
        dY = derivatives_dict[self.loss](y_hat, T); # first factor of each derivative dW(i)
        # calculate the partial derivatives of each layer, and reverse the list (we want to start from the last layer)
        # we get something like partial_activation = {df_last(Z(last))/dZ(last), .., x}
        partial_derivatives = list(derivatives_dict[self.activations[i]](self.partialActivation(X, i)) for i in range(self.activations.shape[0])); # partial derivatives of the net 
        partial_derivatives = partial_derivatives[::-1]; # reverse the list (we will use the net in a reverse fashion)
        partial_derivatives.append(X); # append the last derivative which is X (dWX/dW = X)
        # calculate the partial activation of each function, and reverse the list
        # we get something like partial_activation = {f_last(Z(last)), .., net.W[0]*x}
        partial_activations = list(self.partialActivation(X, i) for i in range(self.activations.shape[0]))[::-1];
        partial_activations.append(X);
        #print("\nPartial activations' functions: \n", partial_activations);
        #print("\nPartial derivatives' functions: \n", partial_derivatives);
        chain = 0;
        #for l in range(len(partial_derivatives)):
        #    print(l, partial_derivatives[l].shape, partial_activations[l].shape);
        for l in range(len(self.W))[::-1]:
            if l != len(self.W)-1:
                #print(l);
                chain = np.dot( self.W[l+1], chain);
                #print(chain.shape)
                chain = np.multiply( chain, partial_derivatives[len(self.W)-l-1]);
                #print(partial_derivatives[len(self.W)-l].shape)
            else:
                #print("bisc",l);
                chain = np.multiply(dY, partial_derivatives[len(self.W)-l-1]);
            dW.append(np.multiply( np.tile(chain, self.W[l].shape[0]).T, partial_activations[len(self.W)-l]) );
            dB = np.append(dB, np.sum(chain));
        #for dw in range(len(dW)):
        #    print("p",dW[dw].shape)
        dW = dW[::-1];
        #print("Weights' updates", dW);
        #print("biases' updates", dB);
        # perform weights update self.W[i] = self.W[i] - l_rate[i]*dY*dW[i]
        for i in range(len(self.W)):
            #print(i);
            self.W[i] -= (net.learning_rate*dW[i]);  # add the learning rate for EACH layer   
            self.Bias[i] -= (net.learning_rate*dB[i]).reshape(1,); 
        return;
        
    # function that performs a step of batch backpropagation of the weights update from the output
    #   (i.e. the loss error) to the varoius weights of the net
    # we use the chain rule to generalize the concept of derivative of the loss wrt the weights
    def batchBackpropagation(self, X, T, batch_size):  
        dW = list(); # list of deltas that are used to calculate weights' update
        dB = np.array([]); # array of deltas that are used to calculate biases' update
        y_hat = (1/batch_size)*np.sum([self.netActivation(x) for x in X]); # prediction of the network
        dY = derivatives_dict[self.loss](y_hat, T); # first factor of each derivative dW(i)
        # calculate the partial derivatives of each layer, and reverse the list (we want to start from the last layer)
        # we get something like partial_activation = {df_last(Z(last))/dZ(last), .., x}
        partial_derivatives = list(derivatives_dict[self.activations[i]]((1/batch_size)*sum([self.partialActivation(x, i) for x in X])) for i in range(self.activations.shape[0])); # partial derivatives of the net 
        partial_derivatives = partial_derivatives[::-1]; # reverse the list (we will use the net in a reverse fashion)
        partial_derivatives.append(sum([x for x in X])); # append the last derivative which is X (dWX/dW = X)
        # calculate the partial activation of each function, and reverse the list
        # we get something like partial_activation = {f_last(Z(last)), .., net.W[0]*x}
        partial_activations = list((1/batch_size)*sum([self.partialActivation(x, i) for x in X]) for i in range(self.activations.shape[0]))[::-1];
        partial_activations.append(X);
        #print("\nPartial activations' functions: \n", partial_activations);
        #print("\nPartial derivatives' functions: \n", partial_derivatives);
        for l in range(len(self.W))[::-1]:
            chain = 0;
            # calculate each dW and append it to the list
            if l != len(self.W)-1: # all the iterations except the first one 
                chain = np.dot(self.W[l+1], chain);
                chain = np.multiply(np.ones(self.W[l].shape), partial_derivatives[len(self.W)-l]);
            else:
                chain = (np.multiply(np.ones(self.W[l].shape), partial_derivatives[0])); # calculate the last derivative and multiply it to the 
            #print(chain);
            dW.append(np.multiply( chain, partial_activations[len(self.W)-l]) );
            dB = np.append(dB, np.sum(chain));
        dW = dW[::-1];
        #print("Weights' updates", dW);
        #print("biases' updates", dB);
        # perform weights update self.W[i] = self.W[i] - l_rate[i]*dY*dW[i]
        for i in range(len(self.W)):
            #print(i);
            self.W[i] -= (net.learning_rate*dY*dW[i]);  # add the learning rate for EACH layer   
            self.Bias[i] -= (net.learning_rate*dY*dB[i]).reshape(1,); 
        return;
    
    
""" Test part """
## create a toy dataset
#X = da.randomData(1000,64);
#Y = da.randomData(1000,10);
#X = da.normalizeData(X); # normalize the input (except for the prediction labels)

net = DeepNet(64, np.array([[128, "sigmoid"], [10, "sigmoid"]]), "L2");
for i in range(len(net.W)):
    net.setWeights(weights_dict['lecun'](net.W[i]), i);
#print("Initial weights ", net.W);
#for n in range(X.shape[1]):
#    net.backpropagation(X[:,n], Y[:,n]);
#X = X.reshape(1000,3,1);
#Y = Y.reshape(1000,1,1);
#for n in range(0, 1000, 100):
#    net.batchBackpropagation(X[n:n+100].reshape(3,100,1), Y[n:n+100].reshape(1,100,1), 100);
#print("Final weights ", net.W);
    
    
""" Test the net with a simple digit recognition test """
import utils_digit_recognition as drec

train_percentage = 90; # percentage of the dataset used for training

digits = drec.load_digits(); # import the dataset

train_size = len(digits.images); # train size is the number of samples in the digits' dataset

images, targets = drec.unison_shuffled_copies(digits.images, digits.target); # shuffle together inputs and supervised outputs

train, test = drec.dataSplit(images, train_percentage);# split train adn test
train_Y, test_Y = drec.dataSplit(targets, train_percentage); # split train and test labels

train_Y = drec.binarization(train_Y); # binarize both the train and test labels
test_Y = drec.binarization(test_Y); # ..


X = train.reshape(train.shape[0], train.shape[1]*train.shape[2]).T;
Y = train_Y;

#X = drec.normalizeData(X);
#Y = drec.normalizeData(Y.T).T;

X_test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).T;
Y_test = test_Y;

#X_test = drec.normalizeData(X_test);
#Y_test = drec.normalizeData(Y_test.T).T;

""" Train with full batch (size of the batch equals to the size of the dataset) """
epochs = 50;
for e in range(epochs):
    print((epochs-e)," epochs left");
    for n in range(X.shape[1]):
        net.backpropagation(X[:,n].reshape(64,1), Y[n].reshape(10,1));

""" test how much we are precise in our prediction """
number_of_errors = 0; # total number of errors on the test set
test_size = X_test.shape[1];
for n in range(X_test.shape[1]):
    if np.argmax(net.netActivation(X_test[:,n].reshape(64,1))) != np.argmax(Y_test[n].reshape(10,1)):
        number_of_errors += 1;
print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");