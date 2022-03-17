from system import System
from neural_network import Network
import matplotlib.pyplot as plt
import numpy as np

class Experiment:

    def __init__(self, m, theta_j, system):
        """
        Experiment class sets up an experiment that determines the resonance
        frequency of a two state system

        Parameters
        ----------
        m: int
            Number of measurements for a given frequency in the discretised
            frequency space.

        theta_j: array_like
            A sequence of the frequencies to test that form the discretised
            frequency space

        system: System Object
            The system that the resonance frequency is being measured.
        """
        self.__m = m
        self.__system = system
        self.__theta_j = theta_j

    def measurement(self):
        """
        Returns array of change of state data for frequencies theta_j
        """
        self.__data = list()
        # loop through frequencies
        for w in self.__theta_j:
            count = 0
            # Find probability of excitation
            prob = self.__system.excitation_probability(w)
            # Test whether state is changed
            for i in range(self.__m):
                r = np.random.uniform(0, 1)
                if r < prob:
                    count += 1
            # Normalise the data
            self.__data.append(count/self.__m)
        x, theory = self.__system.theoretical(10000)
        plt.plot(x, theory, color='black')
        plt.scatter(self.__theta_j, self.__data, color='red', marker='x')
        plt.title("Neural Network Input (Measurement)")
        plt.ylabel(r"Measurement Frequency")
        plt.xlabel(r"Wavelength $\omega_1$")
        plt.legend()
        plt.savefig('Figures/NN_MeasureB.png', dpi=600, bbox_inches='tight')
        plt.show()

    def run(self, no_of_hidden_nodes, learning_rate):
        # Perform experiement (find training distribution)
        self.measurement()
        # Initialise Neural Net with 1 input and N ouput nodes
        NeuralNet = Network(1, len(self.__theta_j), no_of_hidden_nodes,
                            learning_rate)
        prob = self.__system.discrete_theoretical(self.__theta_j)
        # Train the Neural Network
        for i in self.__data:
            n_input = np.array([i])
            NeuralNet.train(n_input, prob)
        output = NeuralNet.run(1)
        plt.plot(self.__theta_j, output, ls='steps', color='black',
                 label=r'Output Nodes $\theta_j$')
        plt.title("Neural Network Output")
        plt.ylabel(r"Posterior Distribution $P(\theta |\mu)$")
        plt.xlabel(r"Wavelength $\omega_1$")
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig('Figures/NN_OutputA.png', dpi=600, bbox_inches='tight')
        plt.show()










