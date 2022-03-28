from neural_network import Network
import matplotlib.pyplot as plt
from function import lorentzian, lorentzian2, gaussian
from scipy.optimize import curve_fit
import numpy as np
from tqdm import tqdm


class Experimentv2:

    def __init__(self, m, k, theta_j, system, plot=False):
        self.__m = m
        self.__k = k
        self.__system = system
        self.__theta_j = theta_j

    def train_NetworkA(self, no_of_output, learning_rate):
        self.__NetworkA = Network(1, len(self.__theta_j), 4,
                                  learning_rate)
        for k in tqdm(range(10)):
            for i in self.__theta_j:
#                training_output = list(self.__system.theoretical(self.__theta_j, i,
#                                                   discrete=True, plot=False)[1])
                training_output = []
                for j in self.__theta_j:
                    if i == j:
                        training_output.append(1)
                    else:
                        training_output.append(0)

                training_input = self.measurement_single(1)
                training_output = np.array([training_output])[0]
                training_input = np.array([training_input])
                print(training_input)
                print(training_output)
                self.__NetworkA.train(training_input, training_output)

    def measurement_single(self, theta):
        """
        Returns array of change of state data for frequencies theta_j
        """

        count = 0
        # Find probability of excitation
        prob = self.__system.excitation_probability(1, theta)
        # Test whether state is changed
        for i in range(self.__m):
            r = np.random.uniform(0, 1)
            if r < prob:
                count += 1
        # Normalise the data
        result = count/self.__m
        return result

    def measurement(self, theta, plot=False):
        """
        Returns array of change of state data for frequencies theta_j
        """
        data = list()

        # loop through frequencies
        for w in self.__theta_j:
            count = 0
            # Find probability of excitation
            prob = self.__system.excitation_probability(w, theta)
            # Test whether state is changed
            for i in range(self.__m):
                r = np.random.uniform(0, 1)
                if r < prob:
                    count += 1
            # Normalise the data
            data.append(count/self.__m)

        if plot is True:
            x, prob = self.__system.theoretical(self.__theta_j, theta,
                                                plot=False)
            plt.plot(x, prob, color='black', label='True Distribution')
            plt.scatter(self.__theta_j, data, color='red', marker='x',
                        label='Sampled Data')
            plt.title("Neural Network Input (Measurement)")
            plt.ylabel(r"Measurement Frequency")
            plt.xlabel(r"Wavelength $\omega/\omega_0$")
            plt.legend()
            plt.savefig('Figures/AII_Sampled_Distribution.png', dpi=600,
                        bbox_inches='tight')
            plt.show()

        return data

    def run(self, learning_rate):
        self.train_NetworkA(100, learning_rate)
        data = self.measurement_single(1)
        output = self.__NetworkA.run(data)
        print(output)
        plt.plot(self.__theta_j, output, color='black', ls='steps-mid',
                 label=r'Output Node $\theta_j$')
        plt.show()
        return output[0][0]



