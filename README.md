## DeepNet
# Basic deep network to evaulate some optimizations techniques in non-shallow architectures

Things implemented so far, 05/12/2017:

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);
  - all from backpropagation to check gradient routine is done by using just numpy
  - add with a human-like line the topology specification (otherwise it's considered fully connected)

- genetic algorithm to find a good network against a train/validation sample of a problem (you can balance between size and accuracy);

- fully connected and locally connected networks;

- sigmoid, relu, leakyrelu, tanh, linear activations and derivatives: easy to extend at your own;


TODO (in order of importance to me):

- improve training of locally connected networks;

- extend the genetic algorithms:
  - learning rate transfer (crossover/mutation): this is **the** hot point;
  - robusteness of the elite by checking it from time to time: tournament among the elements of the elite on the misclassified examples;
  - new crossover/mutation/selection methods.;

- make it work natively with complex functions (mainly for signal processing);
