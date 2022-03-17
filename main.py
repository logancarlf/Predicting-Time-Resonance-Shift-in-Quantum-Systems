from function import DFS
from system_v2 import System
from experiment import Experiment
from neural_network import Network
import numpy as np
import matplotlib.pyplot as plt

Omega = 10

M = 100
mu = 1
std = 0.1
devs = 10
J = 31

# Discretised frequency space
theta_j = DFS(mu, std, devs, J)

# Initiate Simulation and Experiment
qBit = System(Omega)
Bayesian_Estimation = Experiment(M, theta_j, qBit)
Bayesian_Estimation.run(4, 0.6)