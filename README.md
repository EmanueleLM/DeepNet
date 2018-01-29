## DeepNet
# Basic deep network to evaulate some optimizations techniques in non-shallow architectures

## 24/01/2018 NEWS ##

- [COMING NOT SO SOON] parallel training (multiple devices)

## Things implemented so far, 24/01/2018: ##

- customizable deep net (specify with **one** line of code whole the net, from neurons to activations to loss);
  - all from backpropagation to check gradient routine is done by using just numpy
  - add with a human-like line the topology specification (otherwise it's considered fully connected)

- genetic algorithm to find a good network against a train/validation sample of a problem (you can balance between size and accuracy);

- parallel training (multiple processes) to do (weighted) majority vote on classification and regression tasks;

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

While deepplot.netplot provides a (simple and rudimental) way to visualize the net:
```python
import deepplot.netplot as dep
dep.NetPlot(net);
```
![GitHub Logo](/deepplot/images/plot_example.png)

If you want to evolve a population of nets according to a genetic algorithm strategy, use the functions in netgenerator.py.
The best strategy is to look at netgenerator.py last part, that gives you a sketch on how to use all the ga routines in the best way. Anyway there's a fast function to do that, with some parameters that are set by default (but can be changed in invokation phase), here's the syntax:
```python
evolved_population, elite = evolve_population(population_size, epochs, input_size, output_size, selection_type, crossover_type, mutation_type, fully_connected=False, connection_percentage=.5, elite_size=3, crossover_probability=.8, mutation_probability=.05);
```
That can be fastly invoked for a population of 20 nets of the same topology introducted before, through 15 epochs, as:
```python
evolved_population, elite = evolve_population(20, 15, 10, 5, "roulette", "one-point", "random");
```
