# MNIST in Numpy

### About
This is my attempt at making a neural network to classify MNIST images only using numpy (no pytorch).

As of right now, only SGD (Stochastic Gradient Descent) and minibatch gradient descent have both been implemented, though the loss calculation for minibatch (batch > 1) needs to improved right now. To run SGD, just run it with batch_size = 1.

To run, this move the data from the link below into a data/ folder. Afterwards, change the file name accordingly in main.py, and run `python main.py`. Make sure that you have numpy installed as well.

### Data
The data was obtained from a Kaggle, and it can be found [here](https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images).

### Analysis
Obtained a final test accuracy of 0.978 from running SGD once with a learning rate of `1e-2` and 10 epochs. 

