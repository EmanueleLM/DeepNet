# DeepNet
Basic deep network to evaulate the state of the art optimizations in deep learning

Things implemented so far, 05/11/2017:

-customizable deep net (specify with a line of code whole the net, from neurons to activations to loss)

-L1,L2,CrossEntropy losses and derivatives.

-sigmoid, relu, leakyrelu, tanh, linear activations and derivatives (easy to extend at your own)

-tested with 8x8/28x28 handrwritten digits in scikit-learn (few optimizations and hyperparameter tuning, 98%/95% accuracy reached)


TODO (in order of importance to me):

-integrate with a genetic algorithm that finds out the most suitable net for a given problem

-works natively with complex functions

-test against some difficult problem

-implement a good way to manipulate data (data.py is not good)
