# Multi-Task Bayesian Optimization on Extracted Features

All of the experiments are in the `deep-kernel-botorch.py` script.

To run an experiment or change the configurations of an experiment, edit the constants at the beginning of the script.

The "random" experiment generates random data and performs multi-task optimization on random data.

The "elevators" experiment uses the Elevators dataset from UC-Irvine and generates a correlated secondary task.

The "MNIST" experiment optimizes for selecting vectors in the latent space of an MNIST variational autoencoder. You can get some interesting/funny results by optimizing for vectors that have probability mass on the digit "3" and on even numbers.

The "CIFAR" experiment optimizes for selecting vecotrs in the latent space of a CIFAR variational autoencoder. The results here are not great — probably because the latent space for a CIFAR autoencoder is quite high-dimensional.

Admittedly, this is pretty messy and could be improved by passing some of the constants in as command lines and separating things out into separate script files.
