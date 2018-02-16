# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:59:16 2017

@author: Emanuele

Deep network learner with parameter tuning and all sort of optimizations

The starting point is just a feed forward neural net where we add much more layers in depth.
"""

import activations as act
import copy as cp
import derivatives as de
import data as da
import loss as ls
import mask as ma
import numpy as np
import weights as we

# we use a dictionary to handle each layer's activation function: we will need just to know the info contained in
#   DeepNet.layers to invoke the right function!
# Use this struct in this way: 
#   define a function in activations.py (imported let's say as act), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": act.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)
activations_dict = {"relu": act.relu, "sigmoid": act.sigma, "tanh": act.tanh, "leakyrelu": act.leaky_relu, "exp": act.exp, "linear": act.linear};

# the same method is employed for the choice of the loss function
# Use this struct in this way: 
#   define a function in loss.py (imported let's say as ls), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": ls.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)+
loss_dict = {"L1": ls.loss_L1, "L2":ls.loss_L2, "CrossEntropy": ls.loss_cross_entropy};

# the same method is employed for the choice of the derivatives function
# Use this struct in this way: 
#   define a function in derivatives.py (imported let's say as de), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": de.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)
derivatives_dict ={"L1": de.dy_L1, "L2": de.dy_L2, "CrossEntropy": de.dy_cross_entropy, "relu": de.drelu, "leakyrelu": de.dleaky_relu,"sigmoid": de.dsigmoid, "tanh": de.dtanh, "exp": de.dexp, "linear": de.dlinear};

# the same method is employed for the choice of the weights' initialization
# Use this struct in this way: 
#   define a function in weights.py (imported let's say as we), say foo(input)
#   put in the vocabulary the record "name_to_invoke_the_function": we.function_name
#   call the function in this way activation_dictionary["name_to_invoke_the_function"](input)+
weights_dict = {"random": we.random_weights, "uniform":we.uniform_weights, "lecun": we.lecun_weights, "unitary": we.unitary_weights, "flashcard":we.flashcard_weights, "normal": we.normal_weights};

# =============================================================================
#  class that models a deep network with multiple layers and different acrivation functions
#  take as input (the __init__ function):
#      input_size: size of a generic input sample
#      layers: a numpy.array (not a list since we will use numpy.size on it) that contains a number and a string: the former specifies the number of neurons on that layer
#          the latter specifies the activation function (e.g. a valid layers is list([[1,"relu"][5, "sigmoid"]]))
#      loss: the loss function selected for the backpropagation phase
# =============================================================================
class DeepNet(object):
    # define the net architecture:
    #   input_size, the dimension of the input, i.e. the number of input in the first layer
    #   layers, the number of 'hidden' layers, i.e. the number of transformations we apply to the data 
    #   loss, the kind of loss we use to calculate the backprop update
    #   verbose, a boolean (set to False) that prints out a short description of the net
    #   fully_connected, a boolean (set to True) that tells whether the net is fully connected (ffnn) or not
    #   connection_percentage, a floating point number between [0,1] (set to .5) that tells the percentage of connections that are active in an initial non fully connected topology (this one is considere by the code if fully_connected is False)
    def __init__(self, input_size, layers, loss, verbose=False, fully_connected=True, connection_percentage = .5):
        self.weights = list(); # list that contains all the weights in the net (we use a list and np.array for each matrix of weights, for efficiency reasons)
        self.weights.append(np.array(np.zeros([input_size, np.int(layers[0][0])]))); # append the first set of weights
        self.bias = list(); # list that contains all the biases
        self.bias.append(np.array(np.zeros([np.int(layers[0][0]), 1]))); # append the firts set of biases
        self.activations = np.array(layers[:,1]);
        self.loss = loss; # loss of the net
        self.learning_rate = 2e-3; # default learning rate for each iteration phase
        self.momenutum_rate = 0.9; # default momentum rate for the moemntum weights update
        self.fully_connected = fully_connected; # this boolean specify if the net has a fully connected topopolgy 
        self.mask = list(); # list that contains the binary versions of the weights to tell whether a connection exists or not
        self.connection_percentage = connection_percentage; # set the connection percentage (this one is useless if the net is fully connected, anyway it's ok)
        self.accuracy_on_test = 0.; # store the accuracy of the net on the test set (used mainly for performance evaluation and weighted vote)
        # initialize the weights and the biases
        for l in range(len(layers)-1):
            self.weights.append(np.array(np.zeros([np.int(layers[l][0]), np.int(layers[l+1][0])]))); # append all the weights to the list of net's weights
            self.bias.append(np.array(np.zeros([np.int(layers[l+1][0]), 1])));  # append all the biases to the list of net's biases
        # once we have created the weights of the net, just copy (deep) them onto the parameters that are initially the same
        self.weights_old = cp.deepcopy(self.weights);
        self.bias_old = cp.deepcopy(self.bias);
        # non fully connected case and creation of the mask of binary weights with a given percentage of connections turned off
        if fully_connected is False:
            self.fully_connected = False;
            for w in self.weights:
                self.mask.append(np.random.choice([0.,1.], size=w.shape, p=[1.-connection_percentage, connection_percentage]));
        # prints out a short description of the net
        if verbose:
            print("\nNetwork created succesfully!");
            self.explain_net();

    # function that explains the net: this means that it describes the network
    # in terms of input, layers and activation functions
    def explain_net(self):
        print("\nThe network has ", self.weights[0].shape[0], " input(s) and ", len(self.weights), " layers.");
        for l in range(len(self.weights)):
            print("The layer number ", l+1, " has ", self.weights[l].shape[0], " input(s) per neuron, ", self.weights[l].shape[1], " neuron(s) and ", self.activations[l], " as activation function.");
        print("The loss function selected is ", self.loss);
        print("\n\n");
        
    # string method
    def __str__(self):
        net_str = "Network id:"+hex(id(self));
        net_str += "\nNumber of layer(s):"+str(len(self.weights));
        net_str += "\nNumber of input(s) "+str(self.weights[0].shape[0]);
        for i in range(len(self.weights)):
            net_str += "\nLayer #"+str(i+1)+": "+str(self.weights[i].shape[1])+" neuron(s), "+str(self.activations[i])+" as activation";
        net_str += "\nLoss: "+str(self.loss);
        return net_str;
    
    # function that specifies a topology for the net, which is different from a fully connected nn
    # takes as input:
    #   mask, which is a string that is used to specify the topology of the net
    #         take a look at mask.py module in order to understand the (easy) syntax
    def net_topology(self, topology):
        self.mask = ma.Mask(self, topology).weights;
        self.fully_connected = False; # this boolean specifies if the net has a fully connected topology (in this sense if it's not so the backprop and forward change a lot)
        
    # function that activates a single, given layer and, based on its activation function, returns the desired output
    # takes as input
    #   the input X as a vector
    #   the layer where we want the activation to happen
    # returns
    #   the matrix of the activations (even one element for example in single output nn)
    def activation(self, X, layer):
        if self.fully_connected == True:
            return activations_dict[self.activations[layer]](self.weights[layer], X, self.bias[layer]); # activate the activation function of each layer using the vocabulary defined at the beginning            
        else:
            return activations_dict[self.activations[layer]](np.multiply(self.weights[layer], self.mask[layer]), X, self.bias[layer]); # activation with non fully connected topology
    
    # perform activation truncated to a given layer
    # please note that for a L layers nn, we will have L activations Z(1), .., Z(4)
    #   whose dimension is equal to [m,1], where m is the number of neurons the layer ends with 
    def partial_activation(self, X, layer):
        I = X;
        for l in range(layer+1):
            I = self.activation(I, l);
        return I;
    
    # function that activates sequentially all the layers in the net and returns the desired output (a.k.a. "forward")
    # takes as input
    #   the input X as a vector
    # returns
    #   the output vector (even one element for example in single output nn)    
    def net_activation(self, X):
        res = X;
        for l in range(len(self.weights)):
            res = self.activation(res, l);
        return res; 
    
    # function that calculates the loss for a given input, output and a loss function
    # takes as input:
    #   the prediction Y (can be calculated as self.net_activation(input))
    #   the target value 
    # returns:
    #   the loss with the formula specified in self.loss, using the dictionary structure loss_dict to invoke the correct function 
    #       (see above for more info)
    def calculate_loss(self, Y, T):
        return loss_dict[self.loss](Y, T)
    
    # function that performs a step of backpropagation of the weights update from the output
    #   (i.e. the loss error) to the varoius weights of the net
    # we use the chain rule to generalize the concept of derivative of the loss wrt the weights
    # takes as input
    #   X, the input sample X (column vector)
    #   T, the expected output (also as column vector)
    #   update, a boolean (set to True) that sets whether the parameters are updated or just returned
    # returns
    #   dW, dB: the weights/biases updates at this step
    def backpropagation(self, X, T, update=True):  
        dW = list(); # list of deltas that are used to calculate weights' update
        dB = list(); # array of deltas that are used to calculate biases' update
        y_hat = self.net_activation(X); # prediction of the network
        dY = derivatives_dict[self.loss](y_hat, T); # first factor of each derivative dW(i)
        # calculate the partial derivatives of each layer, and reverse the list (we want to start from the last layer)
        # we get something like partial_activation = {df_last(Z(last))/dZ(last), .., x}
        partial_derivatives = list(derivatives_dict[self.activations[i]](self.partial_activation(X, i)) for i in range(self.activations.shape[0])); # partial derivatives of the net 
        partial_derivatives = partial_derivatives[::-1]; # reverse the list (we will use the net in a reverse fashion)
        partial_derivatives.append(X); # append the last derivative which is X (dWX/dW = X)
        # calculate the partial activation of each function, and reverse the list
        # we get something like partial_activation = {f_last(Z(last)), .., net.weights[0]*x}
        partial_activations = list(self.partial_activation(X, i) for i in range(self.activations.shape[0]))[::-1];
        partial_activations.append(X);
        chain = 0;
        for l in range(len(self.weights))[::-1]:
            if l != len(self.weights)-1:
                chain = np.dot( self.weights[l+1], chain);
                chain = np.multiply( chain, partial_derivatives[len(self.weights)-l-1]);
            else:
                chain = np.multiply(dY, partial_derivatives[len(self.weights)-l-1]);
            dW.append(np.multiply( np.tile(chain, self.weights[l].shape[0]).T, partial_activations[len(self.weights)-l]) );
            dB.append(chain);
        dW = dW[::-1]; # now the deltas are ordered as the net goes from left to right (fromt input(s) to ouput(s))      
        dB = dB[::-1];
        #print("Distance between dL/dv and dW.dB, using L2 norm:", self.check_gradient(dW, dB, X, y_hat, T)); # check the gradient's update, the number in the output should be something very low (at least 10**-2)    
        if update is True:
            if self.fully_connected is True:
                self.weights_update(dW, dB); # perform weights update self.weights[i] = self.weights[i] - l_rate[i]*dY*dW[i]
            # backprop with a non fully connected topology
            else:
                for i in range(len(dW)):
                    dW[i] = np.multiply(dW[i], self.mask[i]);
                self.weights_update(dW, dB);
        return dW, dB;     
    
    # function that performs several steps of backpropagation of the weights update from the output
    #   (i.e. the loss error) to the varoius weights of the net
    # we use the chain rule to generalize the concept of derivative of the loss wrt the weights
    # takes as input
    #   X, the input sample X composed by a number of samples (batch)
    #   T, the expected output composed by a number of samples (batch)
    #   batch_size, the size of each batch
    # returns
    #   dW, dB: the weights/biases updates at this step
    def batch_backpropagation(self, X, T, batch_size):
        delta_weights = list([np.zeros(w.shape) for w in self.weights]);
        delta_bias = list([np.zeros(b.shape) for b in self.bias]);
        for i in range(batch_size):
            dW, dB = self.backpropagation(X[:,i].reshape(X[:,i].shape[0], 1), T[:,i].reshape(T[:,i].shape[0], 1), update=False);
            for j in range(len(dW)):
                delta_weights[j] += dW[j]/batch_size;
                delta_bias[j] += dB[j]/batch_size;                
        if self.fully_connected is True:
            self.weights_update(delta_weights, delta_bias); # perform weights update self.weights[i] = self.weights[i] - l_rate[i]*dY*dW[i]
        # backprop with a non fully connected topology
        else:
            for i in range(len(dW)):
                dW[i] = np.multiply(delta_weights[i], self.mask[i]);
            self.weights_update(delta_weights, delta_bias);
    
    # function that performs a step of the ADAGrad backpropagation of the weights update from the output
    #   (i.e. the loss error) to the varoius weights of the net
    # we use the chain rule to generalize the concept of derivative of the loss wrt the weights
    # takes as input
    #   X, the input sample X (column vector)
    #   T, the expected output (also as column vector)
    # returns
    #   dW, dB: the weights/biases updates at this step
    def backpropagation_ada(self, X, T):  
        dW = list(); # list of deltas that are used to calculate weights' update
        dB = list(); # array of deltas that are used to calculate biases' update
        y_hat = self.net_activation(X); # prediction of the network
        dY = derivatives_dict[self.loss](y_hat, T); # first factor of each derivative dW(i)
        # calculate the partial derivatives of each layer, and reverse the list (we want to start from the last layer)
        # we get something like partial_activation = {df_last(Z(last))/dZ(last), .., x}
        partial_derivatives = list(derivatives_dict[self.activations[i]](self.partial_activation(X, i)) for i in range(self.activations.shape[0])); # partial derivatives of the net 
        partial_derivatives = partial_derivatives[::-1]; # reverse the list (we will use the net in a reverse fashion)
        partial_derivatives.append(X); # append the last derivative which is X (dWX/dW = X)
        # calculate the partial activation of each function, and reverse the list
        # we get something like partial_activation = {f_last(Z(last)), .., net.weights[0]*x}
        partial_activations = list(self.partial_activation(X, i) for i in range(self.activations.shape[0]))[::-1];
        partial_activations.append(X);
        chain = 0;
        for l in range(len(self.weights))[::-1]:
            if l != len(self.weights)-1:
                chain = np.dot( self.weights[l+1], chain);
                chain = np.multiply( chain, partial_derivatives[len(self.weights)-l-1]);
            else:
                chain = np.multiply(dY, partial_derivatives[len(self.weights)-l-1]);
            dW.append(np.multiply( np.tile(chain, self.weights[l].shape[0]).T, partial_activations[len(self.weights)-l]) );
            dB.append(chain);
        dW = dW[::-1]; # now the deltas are ordered as the net goes from left to right (fromt input(s) to ouput(s))      
        dB = dB[::-1]; # ..
        #print("Distance between dL/dv and dW.dB, using L2 norm:", self.check_gradient(dW, dB, X, y_hat, T)); # check the gradient's update, the number in the output should be something very low (at least 10**-2)    
        # calculate the weights update according to ADAGrad algorithm
        for n  in range(len(self.weights)):
            # keep trace of the sum of the gradients for ADAGrad algorithm
            self.weights_old[n] += dW[n];
            self.bias_old[n] += dB[n]; 
            # calculate ADAGrad's update
            dW[n] = np.multiply(dW[n], 1/(np.sqrt(np.power(self.weights_old[n], 2))+1e-10)); 
            dB[n] = np.multiply(dB[n], 1/(np.sqrt(np.power(self.bias_old[n], 2))+1e-10));  
        if self.fully_connected == True:
            self.weights_update(dW, dB); # perform weights update self.weights[i] = self.weights[i] - l_rate[i]*dY*dW[i]
        # backprop with a non fully connected topology
        else:
            for i in range(len(dW)):
                dW[i] = np.multiply(dW[i], self.mask[i]);
            self.weights_update(dW, dB);
        return dW, dB;
    
    # function that updates the weights of the net
    # takes as input
    #    dW, the vector that contains all the weights' updates of the net
    #    dB, the same, but for the biases
    def weights_update(self, dW, dB):
        for i in range(len(dW)):
            self.weights[i] -= (self.learning_rate*dW[i]);  # add the learning rate for EACH layer   
            self.bias[i] -= (self.learning_rate*dB[i]); 
            
    # function that updates the weights of the net, with the momentum formula
    # takes as input
    #    dW, the vector that contains all the weights' updates of the net
    #    dB, the same, but for the biases
    def weights_update_momentum(self, dW, dB):
        for i in range(len(dW)):
            self.weights[i] -= (self.learning_rate*dW[i] - self.momenutum_rate*self.weights_old[i]);  # add the learning rate for EACH layer   
            self.bias[i] -= (self.learning_rate*dB[i] - self.momenutum_rate*self.bias_old[i]); 
        self.weights_old = cp.deepcopy(dW); # copy the previous weights
        self.bias_old = cp.deepcopy(dB); # .. and biases
        
    # function that checks if the gradient is computed correctly by comparing it with a 
    #   small perturbation of the loss function
    # here (http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization) is explained the method used
    # practically speaking we check each dW/dB we have calculated with backprop. and we see how much the vector dL/dv (where v is 
    #   one of the parameter of the net) is different from the loss when only v is increased by a small fraction (usually 10**-3)
    # we compare the two vectors dL/dv and dW.dB (concatenated and vectorized), using the 2-norm distance
    # the function takes as input
    #   dW, dB: the changes we calculated for each weight/bias in the backprop phase
    #   X: input sample
    #   Y: prediction of the net with the weights not yet updated
    #   T: prediction (we are in a supervised scenario) 'label' for the given input
    # returns:
    #   the distance between the delta weights/biases vector and the derivative of the loss wrt all the parameters of the net
    def check_gradient(self, dW, dB, X, Y, T):
        epsilon = 10**-3; # the epsioln we use to calculate dL/dv manually
        deltaW = np.array([]); # contains all the weights' update (calculated in backprop phase)
        deltaL = np.array([]); # contains all the derivatives of the loss function wrt a single parameter of the net, evaluated in a specific point (X,Y,T,weights \setminus v,v)
        for i in range(len(self.weights)): # flatten the list of weights/biases
            deltaW = np.append(deltaW, dW[i].flatten() if self.fully_connected is True else np.multiply(dW[i], self.mask[i]).flatten());
        for i in range(len(self.bias)):
            deltaW = np.append(deltaW, dB[i].flatten());            
        for n in range(len(self.weights)): # evaluate the loss with a small perturbation of each parameter of the net, one by one
            for i in range(self.weights[n].shape[0]):
                for j in range(self.weights[n].shape[1]):
                    self.weights[n][i][j] += epsilon*(1. if self.fully_connected is True else self.mask[n][i][j]);                   
                    partialL_plus = np.sum(self.calculate_loss(self.net_activation(X) ,T));
                    self.weights[n][i][j] -= 2*epsilon*(1. if self.fully_connected is True else self.mask[n][i][j]);
                    partialL_minus = np.sum(self.calculate_loss(self.net_activation(X) ,T));
                    self.weights[n][i][j] += epsilon*(1. if self.fully_connected is True else self.mask[n][i][j]);
                    deltaL = np.append(deltaL, (partialL_plus - partialL_minus)/(2*epsilon));
        for n in range(len(self.bias)): # same with the biases vectors
            for j in range(len(self.bias[n])):
                self.bias[n][j] += epsilon;
                partialL_plus = np.sum(self.calculate_loss(self.net_activation(X) ,T));
                self.bias[n][j] -= 2*epsilon;
                partialL_minus = np.sum(self.calculate_loss(self.net_activation(X) ,T));
                self.bias[n][j] += epsilon;
                deltaL = np.append(deltaL, (partialL_plus - partialL_minus)/(2*epsilon));
        # calculate the euclidean distance between deltaW and deltaL
        #print(deltaW.shape, deltaL.shape);
        distance = np.linalg.norm(deltaL - deltaW)/np.linalg.norm(deltaL); # zero valued deltaL (i.e. denominator)
        return distance;  
    
    # function that returns the total number of parameters in the net as sum of the number of weights plus
    # the number of biases
    # returns
    #   params, the number of parameters in of the net
    def number_of_parameters(self):      
        params = 0;
        for w in self.weights:
            params += w.shape[0]*w.shape[1];
        for b in self.bias:
            params += b.shape[0];
        return params
        
""" Test part """
verbose = False;
if verbose:   
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
    #    each input has dimension 10, and each hidden layer has respectively 15, 45, 35 hidden neurons
    #    we want as loss the L1 (lasso)
    #    we just specify:
    #    example_net = DeepNet(10, np.array([[15, "relu"], [45, "relu"], [35, "relu"], [5, "sigmoid"]]), "L1");
    net = DeepNet(1, np.array([[100, "relu"], [75, "relu"], [35, "relu"], [10, "sigmoid"]]), "CrossEntropy", verbose=True); # create the net
    # the first layer is divided in two regions connected, respectively, to each half of the second layer, while the second layer is fully connected to the output
    net.net_topology('layer(1): :32|:5, 33:|6: layer(2): :|: layer(3): :|: layer(4): :|:');
    # set the learning rate
    net.learning_rate = 5e-4;    
    # initialize the weights
    for i in range(len(net.weights)): #initialize the weights
        net.weights[i] = weights_dict['lecun'](net.weights[i]); 
        net.bias[i] = weights_dict['lecun'](net.bias[i]);