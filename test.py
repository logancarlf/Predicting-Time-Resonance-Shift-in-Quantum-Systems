from main import System

B0 = 10  # Tesla
B1 = 1e-5  # Tesla
m = 9.11e-31  # kg
g_factor = 2
alpha0 = 1
beta0 = 0

QBit = System(B0, B1, m, g_factor, alpha0, beta0)
QBit.omega_distribution()
