## DeepNet
# Basic deep network to evaulate the state of the art optimizations in deep learning

Things implemented so far, 30/11/2017:

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);

- L1,L2,CrossEntropy losses and derivatives;

- sigmoid, relu, leakyrelu, tanh, linear activations and derivatives: easy to extend at your own;

- genetic algorithm to find a network with good balance between size and accuracy;

- tested with 8x8/28x28 handrwritten digits in scikit-learn (few optimizations and hyperparameter tuning, 98% accuracy).


TODO (in order of importance to me):

- extend the genetic algorithms' functionalities (new crossover/mutation/selection methods)

- works natively with complex functions

- test against some difficult problem

- good way to manipulate data (data.py is not good)
