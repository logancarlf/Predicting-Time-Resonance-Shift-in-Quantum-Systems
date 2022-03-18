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

        Parameters
        ----------
        omega: float
            Value of scaled frequency for which the excitation probability
            will be calculated.
        '''
        # Define resonance factor as fn/fd
        fn = np.sqrt(self.__Lambda**-2)
        fd = np.sqrt((omega - 1)**2 + self.__Lambda**-2)
        # excitation probability is square of resonance factor
        return (fn/fd) ** 2

    def theoretical(self, theta_j, discrete=False, plot=False):
        '''
        Plots theoretical Resonance Curve for the system

        Parameters
        ----------
        theta_j: array_like
            array of frequencies that the excitation probabilities will be
            calculated for.

        discrete: bool
            If True, the excitation probability will be calculated for every
            value of theta_j, else it will be continuous in the same range.
        '''
        if discrete is False:
            x = np.linspace(theta_j[0], theta_j[-1], 1000)
        else:
            x = theta_j

        # Produce array of probabilities
        prob = list()
        for i in x:
            p = self.excitation_probability(i)
            prob.append(p)

        # Plot continuous theoretical curve
        if plot is True and discrete is False:
            plt.plot(x, prob, color='red', label="Theoretical Curve")
            plt.title("Simulated Resonance Curve")
            plt.ylabel(r"Excitation Probability")
            plt.xlabel(r"Wavelength $\omega/\omega_0$")
            plt.legend()
            plt.savefig('Figures/AI_Resonance_Curve.png', dpi=600,
                        bbox_inches='tight')
            plt.show()

        # Plot discrete distribution (training)
        if plot is True and discrete is True:
            plt.plot(x, prob, color='black', ls='steps-mid',
                     label=r'Output Node $\theta_j$')
            plt.title("Theoretical Training Distribution")
            plt.ylabel(r"Measurement Frequency")
            plt.xlabel(r"Wavelength $\omega/\omega_0$")
            plt.legend()
            plt.savefig('Figures/AIII_Training_Distribution.png', dpi=600,
                        bbox_inches='tight')
            plt.show()

        # Return x-values and associated probablities
        return x, prob