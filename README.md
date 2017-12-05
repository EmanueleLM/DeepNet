## DeepNet
# Basic deep network to evaulate the state of the art optimizations in deep learning

Things implemented so far, 05/12/2017:

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);
  - add with another line the topology specification (otherwise it's considered fully connected)

- L1,L2,CrossEntropy losses and derivatives;

- sigmoid, relu, leakyrelu, tanh, linear activations and derivatives: easy to extend at your own;

- genetic algorithm to find a good network against a train/validation sample of a problem (you can balance between size and accuracy);

- locally connected networks;

- tested with 8x8/28x28 handrwritten digits in scikit-learn (few optimizations and hyperparameter tuning, 98%\96% accuracy).


TODO (in order of importance to me):

- improve training of locally connected networks;

- extend the genetic algorithms:
  - crossover between non-fully connected nets with more than 2 hidden layers;
  - learning rate transfer (crossover/mutation): this is **the** hot point;
  - robusteness of the elite by checking it from time to time: tournament among the elements of the elite on the misclassified examples;
  - new crossover/mutation/selection methods.;

- make it work natively with complex functions;

- test against some difficult problem;

- good way to manipulate data (data.py is not good).
