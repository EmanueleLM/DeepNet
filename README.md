## DeepNet
# Basic deep network to evaulate the state of the art optimizations in deep learning

Things implemented so far, 30/11/2017:

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);

- L1,L2,CrossEntropy losses and derivatives;

- sigmoid, relu, leakyrelu, tanh, linear activations and derivatives: easy to extend at your own;

- genetic algorithm to find a network with balance between size and accuracy;

- tested with 8x8/28x28 handrwritten digits in scikit-learn (few optimizations and hyperparameter tuning, 98%\96% accuracy).


TODO (in order of importance to me):

- extend the genetic algorithms
  - learning rate transfer (crossover/mutation): this is **the** hot point;
  - robusteness of the elite by checking it from time to time;
  - new crossover/mutation/selection methods.

- make it work natively with complex functions

- test against some difficult problem

- good way to manipulate data (data.py is not good)
