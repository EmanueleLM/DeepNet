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
import copy as cp
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
derivatives_dict ={"L1": de.dYL1, "L2": de.dYL2, "CrossEntropy": de.dYCrossEntropy, "relu": de.dRelu, "leakyrelu": de.dLeakyRelu,"sigmoid": de.dSigmoid, "tanh": de.dTanh};

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
        self.W.append(np.array(np.zeros([input_size, np.int(layers[0][0])]))); # append the first set of weights
        self.Bias = list(); # list that contains all the biases
        self.Bias.append(np.array(np.zeros([np.int(layers[0][0]), 1]))); # append the firts set of biases
        self.activations = np.array(layers[:,1]);
        self.loss = loss;
        self.learning_rate = .0005; # default learning rate for each iteration phase
        self.dW_old = list(); # list that contains (usually) the weights of a past iteration: you may use it for the momentum's update
        self.dW_old.append(np.array(np.zeros([input_size, np.int(layers[0][0])])));
        self.dB_old = np.zeros([layers.shape[0], 1]);
        self.momenutum_rate = 0.006; # default momentum rate for the moemntum weights update
        for l in range(len(layers)-1):
            self.W.append(np.array(np.zeros([np.int(layers[l][0]), np.int(layers[l+1][0])]))); # append all the weights to the list of net's weights
            self.Bias.append(np.array(np.zeros([np.int(layers[l+1][0]), 1])));  # append all the biases to the list of net's biases
            self.dW_old.append(np.array(np.zeros([np.int(layers[l][0]), np.int(layers[l+1][0])]))); # this one is to create also a list of the copy of the net's weights
        print("\nNetwork created succesfully!")
        self.explainNet();
        
    # getter and setters for the net elements
    # Setters:
    #   weights
    def setWeights(self, W, layer):
        self.W[layer] = W;
    #   bias
    def setBias(self, B, layer):
        self.Bias[layer] = B;
    # old weights
    def setOldWeights(self, W, layer):
        self.dW_old[layer] = W;
    # old bias
    def setOldBias(self, B, layer):
        self.dB_old[layer] = B;
    # momentum rate
    def setMomentumRate(self, m):
        self.momenutum_rate = m;
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
    def getBias(self, layer):
        return self.Bias[layer];
    #   old weights
    def getOldWeights(self, layer):
        return self.dW_old[layer];
    #   old bias
    def getOldBias(self, layer):
        return self.dB_old[layer];
    # momentum rate
    def getMomentumRate(self, m):
        return self.momenutum_rate;
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
        print("\n\n");
        
    # function that activates a single, given layer and, based on its activation function, returns the desired output
    # takes as input
    #   the input X as a vector
    #   the layer where we want the activation to happen
    # returns
    #   the matrix of the activations (even one element for example in single output nn)
    def activation(self, X, layer):
        Z = np.dot(self.W[layer].T,X) + self.Bias[layer]; # activate (linearly) the input
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
    # takes as input
    #   the input sample X (column vector)
    #   the expected output (also as column vector)
    # returns
    #   dW, dB: the weights/biases updates at this step
    def backpropagation(self, X, T):  
        dW = list(); # list of deltas that are used to calculate weights' update
        dB = list(); # array of deltas that are used to calculate biases' update
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
        chain = 0;
        for l in range(len(self.W))[::-1]:
            if l != len(self.W)-1:
                chain = np.dot( self.W[l+1], chain);
                chain = np.multiply( chain, partial_derivatives[len(self.W)-l-1]);
            else:
                chain = np.multiply(dY, partial_derivatives[len(self.W)-l-1]);
            dW.append(np.multiply( np.tile(chain, self.W[l].shape[0]).T, partial_activations[len(self.W)-l]) );
            dB.append(chain);
        #for dw in range(len(dW)):
        #    print("p",dW[dw].shape);
        dW = dW[::-1]; # now the deltas are ordered as the net goes from left to right (fromt input(s) to ouput(s))      
        dB = dB[::-1];
        #print(dW, "\n\n", dB);
        print(self.checkGradient(dW, dB, X, y_hat, T)); # check the gradient's update
        #print(dW, dB)               
        self.weightsUpdate(dW, dB); # perform weights update self.W[i] = self.W[i] - l_rate[i]*dY*dW[i]
        return dW, dB;
        
    # function that updates the weights of the net
    # takes as input
    #    dW, the vector that contains all the weights' updates of the net
    #    dB, the same, but for the biases
    def weightsUpdate(self, dW, dB):
        for i in range(len(dW)):
            self.W[i] -= (self.learning_rate*dW[i]);  # add the learning rate for EACH layer   
            self.Bias[i] -= (self.learning_rate*dB[i]); 
            
    # function that updates the weights of the net, with the momentum formula
    # takes as input
    #    dW, the vector that contains all the weights' updates of the net
    #    dB, the same, but for the biases
    def weightsUpdateWithMomentum(self, dW, dB):
        #print("magnitude of the update of the first set of weights is ", np.log10(np.abs(np.sum(self.dW_old[0])+ np.sum(dW[0]))));
        for i in range(len(dW)):
            self.W[i] -= (self.learning_rate*dW[i] - self.momenutum_rate*self.dW_old[i]);  # add the learning rate for EACH layer   
            self.Bias[i] -= (self.learning_rate*dB[i] - self.momenutum_rate*self.dB_old[i]).reshape(1,); 
        self.dW_old = cp.deepcopy(dW); # copy the previous weights
        self.dB_old = cp.deepcopy(dB); # .. and biases
        
    # function that checks if the gradient is computed correctly by comparing it with a 
    #   small perturbation of the loss function
    def checkGradient(self, dW, dB, X, Y, T):
        epsilon = 10**-3;
        deltaW = np.array([]);
        deltaL = np.array([]);
        for i in range(len(self.W)):
            deltaW = np.append(deltaW, dW[i].flatten());
        for i in range(len(self.Bias)):
            deltaW = np.append(deltaW, dB[i].flatten());
            
        for n in range(len(self.W)):
            for i in range(self.W[n].shape[0]):
                for j in range(self.W[n].shape[1]):
                    self.W[n][i][j] += epsilon;                   
                    partialL_plus = np.sum(self.calculateLoss(self.netActivation(X) ,T));
                    self.W[n][i][j] -= 2*epsilon;
                    partialL_minus = np.sum(self.calculateLoss(self.netActivation(X) ,T));
                    self.W[n][i][j] += epsilon;
                    deltaL = np.append(deltaL, (partialL_plus - partialL_minus)/(2*epsilon));
        for n in range(len(self.Bias)):
            for j in range(len(self.Bias[n])):
                self.Bias[n][j] += epsilon;
                partialL_plus = np.sum(self.calculateLoss(self.netActivation(X) ,T));
                self.Bias[n][j] -= 2*epsilon;
                partialL_minus = np.sum(self.calculateLoss(self.netActivation(X) ,T));
                self.Bias[n][j] += epsilon;
                deltaL = np.append(deltaL, (partialL_plus - partialL_minus)/(2*epsilon));
        # calculate the euclidean distance between deltaW and deltaL
        #print(deltaW.shape, deltaL.shape);
        #print(deltaW, deltaL);
        distance = np.sqrt(np.sum((np.absolute(np.power(deltaW - deltaL, 2)))))/np.sqrt(np.sum(np.power(deltaW, 2)));
        return distance;
          
    
""" Test part """
## create a toy dataset
#X = da.randomData(1000, 2);
#Y = da.randomData(1000, 1);
#for n in range(X.shape[1]):
#    Y[n] = np.sum(X[:,n]);
#X = da.normalizeData(X); # normalize the input (except for the prediction labels)

# in order to create a net, just specify those few things
#   net = DeepNet(d, np.array([[neurons, act]^(+)]), loss))
#   d is the dimension of the input (x \in R^d for example)
#   [neurons, act]^(+) this is a regexp that indicated that we want at least a layer (of at least one neuron, the output in that case)
#       neurons is a integer that indicates the number of neurons in that layer
#       act is a string that indicates the kind of activation (all of them are displayed in the activations_dict variable on top of this file)
#       loss is a stirng that indicates which loss function we use (all of them are displayed in the loss_dict on the top of this file)
#
# let's make an example:
#    we want to create a 4 layers deep net with all relu in the hidden layers and in the last layer a 5 neurons sigmoid
#    each input has dimension 10, and each hidden layer has respectively 15, 45, 35, 20 hidden neurons
#    we want as loss the L1 (lasso)
#    we just specify:
#    example_net = DeepNet(10, np.array([[15, "relu"], [45, "relu"], [35, "relu"], [5, "sigmoid"]]), "L1");
# net = DeepNet(3, np.array([[5, "sigmoid"], [3, "sigmoid"]]), "L2"); # create a net with this simple syntax
#
## initialize the weights (the way you can initialize them are specified in weights_dict)
#for i in range(len(net.W)):
#    net.setWeights(weights_dict['lecun'](net.W[i]), i);
#    net.setBias(weights_dict['lecun'](net.Bias[i]), i);
#print("\nInitial weights and biases: \n", net.W, net.Bias);
#for n in range(X.shape[1]):
#    net.backpropagation(X[:,n], Y[:,n]);

##print("\nFinal weights and biases: \n", net.W, net.Bias);
#for i in range(20):
#    print(net.netActivation(X[:,i]), Y[:,i]);
net = DeepNet(64, np.array([[128, "sigmoid"], [10, "sigmoid"]]), "L2"); # create the net
for i in range(len(net.W)): #initialize the weights
    net.setWeights(weights_dict['lecun'](net.W[i]), i); 
    
import utils_digit_recognition as drec

train_percentage = 60; # percentage of the dataset used for training
validation_percentage = 20; # this percentage must be lower than the test set, since it's taken directly from it (for the sake of simplicity)

digits = drec.load_digits(); # import the dataset

train_size = len(digits.images); # train size is the number of samples in the digits' dataset

images, targets = drec.unison_shuffled_copies(digits.images, digits.target); # shuffle together inputs and supervised outputs

train, test = drec.dataSplit(images, train_percentage);# split train adn test
train_Y, test_Y = drec.dataSplit(targets, train_percentage); # split train and test labels

validation, test = drec.dataSplit(test, validation_percentage);
validation_Y, test_Y = drec.dataSplit(test_Y, validation_percentage);

train_Y = drec.binarization(train_Y); # binarize both the train and test labels
test_Y = drec.binarization(test_Y); # ..
validation_Y = drec.binarization(validation_Y); # ..


X = train.reshape(train.shape[0], train.shape[1]*train.shape[2]).T;
Y = train_Y;

X = drec.normalizeData(X);
#Y = drec.normalizeData(Y.T).T;

X_test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).T;
Y_test = test_Y;

X_validation = validation.reshape(validation.shape[0], validation.shape[1]*validation.shape[2]).T;
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
        net.backpropagation(X[:,n].reshape(64,1), Y[n].reshape(10,1));
        number_of_errors_validation = 0;
    for n in range(X_validation.shape[1]):
        if np.argmax(net.netActivation(X_validation[:,n].reshape(64,1))) != np.argmax(Y_validation[n].reshape(10,1)):
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
    if np.argmax(net.netActivation(X_test[:,n].reshape(64,1))) != np.argmax(Y_test[n].reshape(10,1)):
        number_of_errors += 1;
print("The error percentage is ", number_of_errors/test_size, ": ", number_of_errors," errors out of ", test_size, " samples on test set.");

