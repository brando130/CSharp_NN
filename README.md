# bnn
Simple neuroevolution of feedforward neural networks through genetic algorithms

This project is my attempt to implement the ideas in Such, et. al 2017 (Uber AI Labs)
https://arxiv.org/abs/1712.06567

The project has minimal concern with training speed. It does have basic optimizations (e.g. using jagged arrays over linked lists) but no multithreading, no GPU math libraries, or the like.

It has two major classes:

NN - Traditional feedforward neural network.
GA - Simple Genetic Algorithm. Creates an initial random population of networks and evolves them by copying and mutating a set number of elites (i.e. the networks with the lowest cost)

The additional classes are 

Training - Imports the training data and defines the cost functions.
NNHelper - Static functions to help with the NN class. (Copy, Print)
MathTools - A number of common neural network functions (ReLU, Sigmoid, Softmax)

