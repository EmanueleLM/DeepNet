# DeepNet
Basic deep network to evaulate the state of the art optimizations in deep learning

Things implemented so far, 22/10/2017:

-customizable deep net (specify with a line of code whole the net, from neurons to activations to loss)

-L1,L2,CrossEntropy losses and derivatives. L1 and L2 have both their complex counterpart

-sigmoid, relu, leakyrelu, tanh activations and derivatives (easy to extend at your own)

-works natively with complex numbers (caveat: note that the function you want to approximate must go from R (real numbers) to C (complex) )

-tested with 8x8 handrwritten digits in scikit-learn (few optimizations and hyperparameter tuning, 98% accuracy reached)


TODO:

-implement a good way to manipulate data (data.py is not good)

-test against some difficult problem

-integrate with a genetic algorithm that finds out the most suitable net for a given problem
