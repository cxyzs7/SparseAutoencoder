__author__ = 'ylou'

from sample_images import sample_images
from display_network import display_network
from sparse_autoencoder_cost import sparse_autoencoder_cost
from math import sqrt
import random
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


def train():
    # STEP 0: Here we provide the relevant parameters values that will
    # allow your sparse autoencoder to get good filters; you do not need to
    # change the parameters below.

    visible_size = 8*8   # number of input units
    hidden_size = 25     # number of hidden units
    sparsity_param = 0.01   # desired average activation of the hidden units.
                        # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                        #  in the lecture notes).
    decay_lambda = 0.0001     # weight decay parameter
    beta = 3            # weight of sparsity penalty term

    # STEP 1: Implement sampleIMAGES
    # After implementing sampleIMAGES, the display_network command should
    # display a random sample of 200 patches from the dataset

    patches = sample_images()
    list = [random.randint(0, patches.shape[0]-1) for i in xrange(64)]
    display_network(patches[list, :], 8)

    # Obtain random parameters theta
    theta = initialize_parameters(hidden_size, visible_size)

    # STEP 2: Implement sparseAutoencoderCost
    #
    #  You can implement all of the components (squared error cost, weight decay term,
    #  sparsity penalty) in the cost function at once, but it may be easier to do
    #  it step-by-step and run gradient checking (see STEP 3) after each step.  We
    #  suggest implementing the sparseAutoencoderCost function using the following steps:
    #
    #  (a) Implement forward propagation in your neural network, and implement the
    #      squared error term of the cost function.  Implement backpropagation to
    #      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking
    #      to verify that the calculations corresponding to the squared error cost
    #      term are correct.
    #
    #  (b) Add in the weight decay term (in both the cost function and the derivative
    #      calculations), then re-run Gradient Checking to verify correctness.
    #
    #  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
    #      verify correctness.
    #
    #  Feel free to change the training settings when debugging your
    #  code.  (For example, reducing the training set size or
    #  number of hidden units may make your code run faster; and setting beta
    #  and/or lambda to zero may be helpful for debugging.)  However, in your
    #  final submission of the visualized weights, please use parameters we
    #  gave in Step 0 above.

    cost, grad = sparse_autoencoder_cost(theta, visible_size, hidden_size, decay_lambda, sparsity_param, beta, patches)




if __name__ == "__main__":
    train()