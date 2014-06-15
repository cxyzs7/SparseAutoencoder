from random import randint
from math import sqrt

import numpy as np


def initialize_parameters(hidden_size, visible_size):
    # Initialize parameters randomly based on layer sizes.
    r = sqrt(6) / sqrt(hidden_size+visible_size+1)   # we'll choose weights uniformly from the interval [-r, r]
    w1 = np.random.rand(hidden_size, visible_size) * 2 * r - r
    w2 = np.random.rand(visible_size, hidden_size) * 2 * r - r

    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((visible_size, 1))

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    theta = np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    return theta


# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sparse_autoencoder_cost_and_grad(theta, visible_size, hidden_size, decay_lambda, sparsity_param, beta, data):
    # visible_size: the number of input units (probably 64)
    # hidden_size: the number of hidden units (probably 25)
    # decay_lambda: weight decay parameter
    # sparsity_param: The desired average activation for the hidden units (denoted in the lecture
    #                           notes by the greek alphabet rho, which looks like a lower-case "p").
    # beta: weight of sparsity penalty term
    # data: Our 10000x64 matrix containing the training data.  So, data(i,:) is the i-th training example.
    
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    num_combinations = visible_size*hidden_size
    w1 = theta[0:num_combinations].reshape((hidden_size, visible_size))
    w2 = theta[num_combinations:2*num_combinations].reshape((visible_size, hidden_size))
    b1 = theta[2*num_combinations:2*num_combinations+hidden_size]
    b2 = theta[2*num_combinations+hidden_size:]
    
    # Cost and gradient variables (your code needs to compute these values).
    # Here, we initialize them to zeros.
    cost = 0
    w1grad = np.zeros_like(w1)
    w2grad = np.zeros_like(w2)
    b1grad = np.zeros_like(b1)
    b2grad = np.zeros_like(b2)

    #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    #
    # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
    # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
    # [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
    # of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    #
    # Stated differently, if we were using batch gradient descent to optimize the parameters,
    # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
    #
    
    
    
    
    
    
    
    

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.

    grad = np.concatenate((w1grad.flatten(), w2grad.flatten(), b1grad.flatten(), b2grad.flatten()))

    return cost, grad


def sparse_autoencoder_cost(theta, visible_size, hidden_size, decay_lambda, sparsity_param, beta, data):
    return sparse_autoencoder_cost_and_grad(theta, visible_size, hidden_size,
                                            decay_lambda, sparsity_param, beta, data)[0]