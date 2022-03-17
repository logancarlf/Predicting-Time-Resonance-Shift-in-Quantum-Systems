from main import System
from neural_network import Network
import numpy as np


B0 = 10  # Tesla
B1 = 1e-5  # Tesla
m = 9.11e-31  # kg
g_factor = 2
alpha0 = 1
beta0 = 0

devs = 10

QBit = System(B0, B1, m, g_factor, alpha0, beta0)
QBit.omega_distribution()
data = QBit.measurement(50)
prob = QBit.theoretical(31)

no_of_in_nodes = 1
no_of_out_nodes = 31
no_of_hidden_nodes = 4
learning_rate = 0.6

res_mean = QBit.resonance_mean()
res_std = QBit.resonance_std()
theta_grid = np.linspace(res_mean - devs * res_std, res_mean + devs * res_std,
                             no_of_out_nodes)


NeuralNet = Network(no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes,
                 learning_rate)

# train
for i in data:
    n_input = np.array([i])
    NeuralNet.train(n_input, prob)

measure = QBit.measurement(1)
output = NeuralNet.run(measure)
output_np = np.array(output)
output = output_np.T[0]
width = theta_grid[1] - theta_grid[0]

#moves data
data_move()

plt.plot(theta_grid, output, ls='steps', color='black',
         label=r'Output Nodes $\theta_j$')
plt.title("Neural Network Output")
plt.ylabel(r"Posterior Distribution $P(\theta |\mu)$")
plt.xlabel(r"Wavelength $\omega_1$")
plt.ylim(0, 1)
plt.legend()
plt.show()





