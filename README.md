# MNIST in Numpy

# About
This is my attempt at making a neural network to classify MNIST images only using numpy (no pytorch).

As of right now, only SGD (Stochastic Gradient Descent) has been implemented, but I am currently writing 
the enable minibatch as well.

# Data
The data was obtained from a Kaggle, and it can be found [here](https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images). The way I formatted my files was with a data/ folder, and I stored the data there. 

# Analysis
Obtained a final test accuracy of 0.978 from running SGD once with a learning rate of `1e-2` and 10 epochs. 