import numpy as np

e = 1.602176634e-19
i = (-1) ** 0.5

def DFS(mu, sigma, devs, N):
    x = np.linspace(mu - devs * sigma, mu + devs * sigma, N)
    return x


def lorentzian(x, a, x_0, gamma):
    return (a * (1/np.pi) * (0.5*gamma))/((x-x_0)**2 + (0.5*gamma)**2)