import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm


class Network:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes,
                 learning_rate):
        """
        Class object creates a neural network to compute the Bayesian
        parameter estimation
        """
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuples, lists or ndarrays
        """
        # make sure that the vectors have the right shape
        input_npvector = np.array(input_vector)
        input_vector = input_vector.reshape(input_npvector.size, 1)
        target_npvector = np.array(target_vector)
        target_vector = target_npvector.reshape(target_npvector.size, 1)

        output_vector_hidden = activation_function(
                self.weights_in_hidden @ input_vector)
        output_vector_network = activation_function(
                self.weights_hidden_out @ output_vector_hidden)

        output_error = target_vector - output_vector_network
        tmp = output_error * output_vector_network \
            * (1.0 - output_vector_network)
        self.weights_hidden_out += self.learning_rate \
            * (tmp @ output_vector_hidden.T)

    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        """
        # Turn the input vector into a column vector:
        input_vector = np.array(input_vector, ndmin=2).T
        # activation_function() implements the expit function,
        # which is an implementation of the sigmoid function:
        input_hidden = activation_function(
                self.weights_in_hidden @   input_vector)
        output_vector = activation_function(
                self.weights_hidden_out @ input_hidden)
        return output_vector


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
