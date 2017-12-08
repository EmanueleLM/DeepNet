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

## Create a network ##
Let's suppose we want to build a 3 layers (input + 2 hidden layers) neural net, with sigmoid in the first hidden layer,
exp in the second layer and a leakyrelu in the last layer (let's suppose its a good architecture :-S ). We want an input which is 
10-dimensional, 35 neurons in the first hidden layer, 40 neurons in the second hidden layer and 5 neurons in the output layer.
We will use the L2 error measure.
We will have:
```python
net = DeepNet(10, [[35, "sigmoid"], [40, "exp"], [5, "leakyrelu"]], "L2");
```

We want to specify the topology of the net, in such a way that the input is connected in this way:
- the inputs from the first to the fifth are connected to the first 20 neurons of the first hidden layer
- the 6th and 7th inputs are connected, respectively, to the 21th and 22th neurons of the first hidden layer; 
- the rest of the inputs are fully connected to the rest of the neurons of the first hidden layer;
- the rest of the net is fully connected (just to have few text to read in this tutorial, the way you connect the other layers is very the same as what I described above).
```python
net.netTopology('layer(1): :5|:20, 6|21, 7|22, 8:10|23:35 layer(2): :35|:40 layer(3): :40|:5'); 
```
