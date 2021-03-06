## DeepNet
# Basic deep network to evaulate some optimizations techniques in non-shallow architectures

# Coming Soon
genetic-ng module (ok I will call it 'genetic' and substitute the old one).
- this one will be a full working module with several operators and operations that are fast and reliable;
  - possibility to use the full net object or just a string to do mutation and crossover;
  - Darwin vs. Lamarck evolutions' models (should we transmit something the net has learnt during its life?). 
- will be used to test a fresh new genetic method based on KC/LC complexity among each layer.

## Things implemented so far, 30/01/2018: ##

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);
  - all from backpropagation to check gradient routine is done by using just numpy
  - add with a human-like line the topology specification (otherwise it's considered fully connected)

- genetic algorithm to find a good network against a train/validation sample of a problem (you can balance between size and accuracy);

- train multiple networks using parallel processes and do classification/regression by using multiple (weighted) majority vote;

- fully connected and locally connected networks;

- sigmoid, relu, leakyrelu, tanh, linear activations and derivatives: easy to extend at your own;

- simple plot system.


TODO (in order of importance to me):

- improve training of locally connected networks;

- extend the genetic algorithms:
  - learning rate transfer (crossover/mutation): this is **the** hot point;
  - robusteness of the elite by checking it from time to time: tournament among the elements of the elite on the misclassified examples;
  - new crossover/mutation/selection methods.
 
- make it work natively with complex functions (mainly for signal processing).

## Create a network ##
Let's suppose we want to build a 3 layers (input + 2 hidden layers) neural net, with sigmoid in the first hidden layer,
exp in the second layer and a leakyrelu in the last layer (let's suppose its a good architecture :-S ). We want an input which is 
10-dimensional, 35 neurons in the first hidden layer, 40 neurons in the second hidden layer and 5 neurons in the output layer.
We will use the L2 error measure.
We will have:
```python
net = DeepNet(10, np.array([[35, "sigmoid"], [40, "exp"], [5, "leakyrelu"]]), "L2");
```

We want to specify the topology of the net, in such a way that the input is connected in this way:
- the inputs from the first to the fifth are connected to the first 20 neurons of the first hidden layer
- the 6th and 7th inputs are connected, respectively, to the 21th and 22th neurons of the first hidden layer; 
- the rest of the inputs are fully connected to the rest of the neurons of the first hidden layer;
- the rest of the net is fully connected (just to have few text to read in this tutorial, the way you connect the other layers is very the same as what I described above).
```python
net.net_topology('layer(1): :5|:20, 6|21, 7|22, 8:|23: layer(2): :|: layer(3): :|:'); 
```

Alternatively we can build up a network by first specifying the first layer (at least one) as stated in the previous snippet of code, then add the others, each one with its specific topology (fully or non fully connected).
```python
net = DeepNet(10, np.array([[35, "sigmoid"]), "L2");
# add a block non-fully connected
net.add_block(40, "exp", fully_connected=False, connection_percentage=.75);
net.add_block(5, "leakyrelu");
```

And we can remove also blocks from our net, by simply using this function:
```python
# remove the last one
net.remove_block(-1);
# remove the first layer of weights
net.remove(0); 
# remove the block after the input
net.remove_block(1); 
```

While deepplot.netplot provides a (simple and rudimental) way to visualize the net:
```python
import deepplot.netplot as dep
dep.NetPlot(net);
```
![GitHub Logo](/deepnet/deepplot/images/plot_example.png)

If you want to evolve a population of nets according to a genetic algorithm strategy, use the functions in netgenerator.py.
Here's the syntax you should use just for the evolution part: soon I will a more complex example.
```python
# let's suppose we have 4 nets, net1, net2, net3, net4
nets = [net1, net2, net3, net4];
evolve(nets, selection="rank", crossover="one-point", mutation="random");
```
