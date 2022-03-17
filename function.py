import numpy as np

e = 1.602176634e-19
i = (-1) ** 0.5

def DFS(mu, sigma, devs, N):
    x = np.linspace(mu - devs * sigma, mu + devs * sigma, N)
    return x