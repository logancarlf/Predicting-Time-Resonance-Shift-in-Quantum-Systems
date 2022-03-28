import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

e = 1.602176634e-19
i = (-1) ** 0.5


def DFS(mu, sigma, devs, N):
    x = np.linspace(mu - devs * sigma, mu + devs * sigma, N)
    return x


def lorentzian(x, a, x_0, gamma, c):
    return (a * (1/np.pi) * (0.5*gamma))/((x-x_0)**2 + (0.5*gamma)**2) + c


def lorentzian2(x, Lambda):
        fn = np.sqrt(Lambda**-2)
        fd = np.sqrt((x - 1)**2 + Lambda**-2)
        # excitation probability is square of resonance factor
        return (fn/fd) ** 2


def gaussian(x, a, x_0, sigma, c):
    return a * np.exp(-(x-x_0)**2/(2*sigma**2)) + c


class resonance:

    def __init__(self, start, end, dt):
        np.random.seed(3124523)
        self.__t = np.arange(start, end, dt)
        self.__Lambda_t = [1]
        while len(self.__Lambda_t) != len(self.__t):
            new_Lambda = self.__Lambda_t[-1] + 0.001
            self.__Lambda_t.append(new_Lambda)
        plt.plot(self.__t, self.__Lambda_t, color='black')
        plt.xlabel("Time $t$")
        plt.ylabel(r"Resonance Factor $\Lambda$")
        plt.title("Resonance Frequency Shift")
        plt.savefig('Figures/BI,_Resonance_Shift.png', dpi=600,
                        bbox_inches='tight')
        plt.show()

    def shift(self, t):
        interpolation = interp1d(self.__t, self.__Lambda_t)
        return interpolation.__call__(t)

    def plot(self, x, y, stds):
        measurement = []
        for i in range(10):
            m = self.shift(i)
            measurement.append(m)
        print("Hello", measurement)
        t = np.arange(0, 10, 1)
        plt.plot(t, measurement, marker='x', color='red', zorder=10,
                    linestyle='dashed')
        plt.plot([t[-1], x[10]], [measurement[-1], y[10]], color='red', zorder=10,
                    linestyle='dashed')
        plt.plot(self.__t, self.__Lambda_t, color='lightgrey',
                 label='Resonance Shift')
        plt.plot(x[10:], y[10:],color='red', marker='x',linestyle='dashed',
                 label='Neural Network Prediction')
        plt.errorbar(x[10:], y[10:], yerr=np.sqrt(stds[10:]), ls='none',
                     color='red', capsize=3, zorder=9)
        plt.title("Network Prediction [L.R. = $10^{-3}$]")
        plt.xlabel("Time $t$")
        plt.ylabel(r"Resonance Factor $\Lambda$")
        plt.legend()
        plt.savefig('Figures/BIV_Prediction.png', dpi=600,
                        bbox_inches='tight')
        plt.show()









