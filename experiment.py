from system import System
from neural_network import Network
import matplotlib.pyplot as plt
from function import lorentzian, gaussian
from scipy.optimize import curve_fit
import numpy as np

class Experiment:

    def __init__(self, m, theta_j, system, plot=False):
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
        self.__plot = plot
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

        if self.__plot is True:
            x, prob = self.__system.theoretical(self.__theta_j,
                                                plot=self.__plot)
            plt.plot(x, prob, color='black', label='True Distribution')
            plt.scatter(self.__theta_j, self.__data, color='red', marker='x',
                        label='Sampled Data')
            plt.title("Neural Network Input (Measurement)")
            plt.ylabel(r"Measurement Frequency")
            plt.xlabel(r"Wavelength $\omega/\omega_0$")
            plt.legend()
            plt.savefig('Figures/AII_Sampled_Distribution.png', dpi=600, bbox_inches='tight')
            plt.show()

    def run(self, no_of_hidden_nodes, learning_rate):
        # Perform experiement (find training distribution)
        self.measurement()
        # Initialise Neural Net with 1 input and N ouput nodes
        NeuralNet = Network(1, len(self.__theta_j), no_of_hidden_nodes,
                            learning_rate)
        # Produce training distirbution
        prob = self.__system.theoretical(self.__theta_j, discrete=True,
                                         plot=self.__plot)[1]
        # Train the Neural Network
        for i in self.__data:
            n_input = np.array([i])
            NeuralNet.train(n_input, prob)
        output = NeuralNet.run(1).flatten()

        # Fit Lorentzian to find resonance frequency
        guess = [0.2, 1, 0.2, 0.2]
        Lfit, Lcov = curve_fit(lorentzian, self.__theta_j, output, p0=guess,
                               maxfev=10000)
        # Fit Gaussian to find resonance frequency
        guess = [0.55, 1, -0.01, 0.2]
        Gfit, Gcov = curve_fit(gaussian, self.__theta_j, output, p0=guess,
                               maxfev=10000)

        if self.__plot is True:
            x = np.linspace(self.__theta_j[0], self.__theta_j[-1], 1000)
            plt.plot(x, lorentzian(x, *Lfit), linestyle='dashed', color='red',
                     label='Lorentzian Fit')
            plt.plot(x, gaussian(x, *Gfit), linestyle='dashed', color='blue',
                     label='Gaussian Fit')
            plt.plot(self.__theta_j, output, ls='steps-mid', color='black',
                     label=r'Output Nodes $\theta_j$')
            plt.title("Neural Network Output")
            plt.ylabel(r"Posterior Distribution $P(\theta |\mu)$")
            plt.xlabel(r"Wavelength $\omega /\omega_0$")
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig('Figures/AIV_Output_Distibrution.png', dpi=600,
                        bbox_inches='tight')
            plt.show()

        print("Lorentzian Result:", Lfit[1], "+/-", np.sqrt(Lcov[1][1]))
        print("Gaussian Result:", Gfit[1], "+/-", np.sqrt(Gcov[1][1]))










