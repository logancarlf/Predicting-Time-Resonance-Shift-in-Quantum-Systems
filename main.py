from function import DFS
from system import System
from experiment import Experiment
from neural_network import Network
import numpy as np
import matplotlib.pyplot as plt

B0 = 10  # Tesla
B1 = 1e-5  # Tesla
m = 9.11e-31  # kg
g_factor = 2
alpha0 = 1
beta0 = 0

M = 100
mu = 1758701025246.9812
std = 1758701.0252469813
devs = 10
J = 31

# Discretised frequency space
theta_j = DFS(mu, std, devs, J)

# Initiate Simulation and Experiment
qBit = System(B0, B1, m, g_factor, alpha0, beta0)
Bayesian_Estimation = Experiment(M, theta_j, qBit)
Bayesian_Estimation.run(4, 0.6)