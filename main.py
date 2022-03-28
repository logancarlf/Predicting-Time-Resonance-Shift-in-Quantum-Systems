from function import DFS, resonance
from system_v3 import System
from experiment import Experiment, Experiment_Time
from Experiment_v2 import Experimentv2
import numpy as np

Omega = 10

M = 100
K = 10
mu = 1

std = 0.1
devs = 20
J = 31

# Discretised frequency space
theta_j = DFS(mu, std, devs, J)

# Initiate Simulation and Experiment
#qBit = System(Omega)
#Bayesian_Estimation = Experiment(M, theta_j, qBit, plot=True)
#Bayesian_Estimation.run(4, 0.6)

# Time-Dependent System
lambda_t = resonance(0, 20.1, 0.01)

#qBit = System_Time_Dependent(lambda_t.shift, 5)
#Bayesian_Estimation = Experiment_Time(M, K, theta_j, qBit)
#Bayesian_Estimation.run_sim(1, 10)
#
#t = np.arange(0, 20, 1)
#results = list()
#stds = list()
#for i in t:
#    x, std, = Bayesian_Estimation.expectation_value(i, plot=True)
#    results.append(x)
#    stds.append(std)
#
#lambda_t.plot(t, results, stds)


qBit = System(5)
Bayesian_Estimation = Experimentv2(M, K, theta_j, qBit)
Bayesian_Estimation.run(0.1)

#x = np.arange(-1, 1, 0.1)
#g = []
#for i in x:
#    out = Bayesian_Estimation.run(i)
#    g.append(out)
#
#plt.plot(x, g)
