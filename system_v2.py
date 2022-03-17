import function as const
import scipy.integrate as sp
import matplotlib.pyplot as plt
import numpy as np


class System:

    def __init__(self, Lambda):
        self.__Lambda = Lambda

    def excitation_probability(self, omega):
        '''
        Calculates the maximum probability of changing the state of the system
        for a given perturbation frequency omega
        '''
        # Define resonance factor as fn/fd
        fn = np.sqrt(self.__Lambda**-2)
        fd = np.sqrt((omega - 1)**2 + self.__Lambda**-2)
        # excitation probability is square of resonance factor
        return (fn/fd) ** 2

    def theoretical(self, theta_cont):
        '''
        Plots theoretical Resonance Curve for the system
        '''
        x = np.linspace(theta_cont[0], theta_cont[-1], 1000)
        prob = list()
        for i in x:
            p = self.excitation_probability(i)
            prob.append(p)
        plt.plot(x, prob, color='red')
        plt.title("Neural Network Input (Measurement)")
        plt.ylabel(r"Measurement Frequency")
        plt.xlabel(r"Wavelength $\omega/\omega_0$")
        plt.legend()
        plt.savefig('Figures/NN_MeasureB.png', dpi=600, bbox_inches='tight')
        plt.show()
        return x, prob