import function as const
import scipy.integrate as sp
import matplotlib.pyplot as plt
import numpy as np


class System:

    def __init__(self, B0, B1, m, g_factor, alpha0, beta0, Omega):
        '''
        Class object simulates a 2-state Quantum System that undergoes a
        magnetic field perturbation based on the numerous intrinsic properties
        of the system.

        Parameters
        ----------
        B0: Strength of the static Magnetic Field.

        B1: Strength of the magnetic field of the electromagnetic perturbation
        (B1 << B0)

        m: Mass of the system

        g_factor: Dimensionless Magnetic moment parameter (usually ~2)

        alpha0: Magnitude if up-state in vectorspace

        beta0: Magnitude if down-state in vectorspace
        '''
        # define intrinsic system properties
        self.__g_factor = g_factor
        self.__alpha0 = alpha0
        self.__beta0 = beta0
        self.__B0 = B0
        self.__B1 = B1
        self.__m = m

        self.__Omega = Omega

        # derive and define extrinsic system properties
        self.__gyro_ratio = (const.e * self.__g_factor) / (2 * self.__m)
        self.__omega0 = self.__B0 * self.__gyro_ratio
        self.__omega1 = self.__B1 * self.__gyro_ratio

#    def schrodinger(self, Psi, t):
#        '''
#        Coupled ODE Schrodinger Equations for the variation of the vectorspace
#        coefficients with time (for given omega)
#        '''
#        # Extract coefficeints from eigenvector
#        alpha, beta = Psi
#        # Change in alpha in time
#        d_alpha_d_t = 0.5 * 1j * (self.__omega0 * alpha + self.__omega1 * beta
#                                  * np.exp(1j * self.__omega * t))
#        # Change in beta in time
#        d_beta_d_t = 0.5 * 1j * (-self.__omega0 * beta + self.__omega1 * alpha
#                                 * np.exp(-1j * self.__omega * t))
#        # Return derivative vector
#        return d_alpha_d_t, d_beta_d_t
#
#    def wavefunction(self, t, omega):
#        fn = np.sqrt(self.__omega1**2)
#        fd = np.sqrt((self.__omega-self.__omega0)**2 + self.__omega1**2)
#        alpha = (self.__alpha0 * (np.cos((self.__omega1 * t * fd)/(2 * fn))))
#
#        return alpha, beta
#
    def excitation_probability(self, omega):
        '''
        Calculates the maximum probability of changing the state of the system
        for a given perturbation frequency omega
        '''
        # Define resonance factor as fn/fd
        fn = np.sqrt(self.__omega1**2)
        fd = np.sqrt((omega - self.__omega0)**2 + self.__omega1**2)
        # excitation probability is square of resonance factor
        return (fn/fd) ** 2

    def omega_distribution(self):
        '''
        Calculates the values of the maximum excitation energy for a set of
        perturbation frequencies and plots the resonance curve
        '''
        # Number of stds on either side of the mean
        dev = 10
        # Set distribution parameters
        mu, sigma = self.__omega0, self.__omega1
        # Get array of omega values
        omega_array = np.arange(mu - dev * sigma, mu + dev * sigma, sigma/1000)
        # Get values of the max excitation probability
        excite_p = list()
        for omega in omega_array:
            P_omega = self.excitation_probability(omega)
            excite_p.append(P_omega)
        # Plot resonance curve
        plt.plot(omega_array, excite_p,color='black')
        plt.title("Resonance Curve")
        # Axis Labels
        plt.xlabel(r"Perturbation Frequency $\omega$ / $s^{-1}$")
        plt.ylabel(r"Maximum Excitation Probability $P_{max}$")
        plt.grid()
        plt.savefig('Figures/Res_Curve.png', dpi=600, bbox_inches='tight')
        plt.show()

    def excitation_prob_2(self, omega):
        '''
        Calculates the maximum probability of changing the state of the system
        for a given perturbation frequency omega
        '''
        # Define resonance factor as fn/fd
        fn = np.sqrt(self.__Omega**-2)
        fd = np.sqrt((omega - 1)**2 + self.__Omega**-2)
        # excitation probability is square of resonance factor
        return (fn/fd) ** 2

    def theoretical_2(self):
        x = np.linspace(-4, 6, 1000)
        prob = list()
        for i in x:
            p = self.excitation_prob_2(i)
            prob.append(p)
        plt.plot(x, prob)
        plt.show()

#    def theoretical(self, N):
#        dev = 10
#        mu, sigma = self.__omega0, self.__omega1
#        self.__grid = np.linspace(mu - dev * sigma, mu + dev * sigma, N)
#        prob = list()
#        for i in self.__grid:
#            p = self.excitation_probability(i)
#            prob.append(p)
#        width = self.__grid[1] - self.__grid[0]
#        plt.plot(self.__grid, prob, ls='steps', color='red',
#                 label=r'Resonance Curve')
#        plt.title("Excitation Probability")
#        plt.ylabel(r"Posterior Distribution $P(\theta |\mu)$")
#        plt.xlabel(r"Wavelength $\omega_1$")
#        plt.legend()
#        plt.savefig('Figures/NN_TrainA.png', dpi=600, bbox_inches='tight')
#        plt.show()
#        return self.__grid, prob
#
#    def discrete_theoretical(self, theta_j):
#        prob = list()
#        for i in theta_j:
#            p = self.excitation_probability(i)
#            prob.append(p)
#        width = theta_j[1] - theta_j[0]
#        plt.plot(theta_j, prob, ls='steps', color='black',
#                 label=r'Output Nodes $\theta_j$')
#        plt.title("NN Training Distribution")
#        plt.ylabel(r"Posterior Distribution $P(\theta |\mu)$")
#        plt.xlabel(r"Wavelength $\omega_1$")
#        plt.legend()
#        plt.savefig('Figures/NN_TrainA.png', dpi=600, bbox_inches='tight')
#        plt.show()
#        return prob
#
#    def measurement(self, N):
#        # Number of stds on either side of the mean
#        dev = 20
#        # Set distribution parameters
#        mu, sigma = self.__omega0, self.__omega1
#        # array to store perturbation frequencies
#        excitation_frequencies = list()
#        while len(excitation_frequencies) < N:
#            # Generate random number
#            p = np.random.uniform(mu - dev * sigma, mu + dev * sigma)
#            q = np.random.uniform(0, 1)
#            # Calculate excitation probability
#            prob = self.excitation_probability(p)
#            if q < prob:
#                excitation_frequencies.append(p)
#        # Plot histogram
#        plt.hist(excitation_frequencies, bins=31, color='black',
#                 label=r'Input Nodes $\theta_i$')
#        plt.title("Neural Network Input (Measurement)")
#        plt.ylabel(r"Measurement Frequency")
#        plt.xlabel(r"Wavelength $\omega_1$")
#        plt.legend()
#        plt.savefig('Figures/NN_MeasureA.png', dpi=600, bbox_inches='tight')
#        plt.show()
#        return excitation_frequencies
#
#    def resonance_mean(self):
#        return self.__omega0
#
#    def resonance_std(self):
#        return self.__omega1
#
#    def time(self):
#        return self.__time





#%%

#    def schrodinger(self, Psi, t):
#        alpha, beta = Psi
#        d_alpha_d_t = 0.5 * 1j * (self.__omega0 * alpha + self.__omega1 * beta * np.exp(1j * self.__omega * t))
#        d_beta_d_t = 0.5 * 1j * (-self.__omega0 * beta + self.__omega1 * alpha * np.exp(-1j * self.__omega * t))
#        return d_alpha_d_t, d_beta_d_t
#
#    def simulate(self):
#        print(self.__omega0)
#        print(self.__omega1)
#        self.__omega = self.__omega0
#        t_max = 20 / self.__omega1
#        self.__time = np.arange(0, t_max, t_max/100)
#        dt = self.__time[1] - self.__time[0]
#        Psi = list()
#        Psi.append(np.array([self.__alpha0, self.__beta0]))
#        for i in range(len(self.__time)-1):
#            print(Psi[i])
#            f_a = self.schrodinger(Psi[i], self.__time[i])
#            f_b = self.schrodinger(Psi[i] + np.array(f_a) * dt * 0.5, self.__time[i] + dt * 0.5)
#            f_c = self.schrodinger(Psi[i] + np.array(f_b) * dt * 0.5, self.__time[i] + dt * 0.5)
#            f_d = self.schrodinger(Psi[i] + np.array(f_c) * dt, self.__time[i] + dt)
#            u_n_1 = Psi[i] +1/6 * (np.array(f_a) + 2*np.array(f_b) + 2*np.array(f_c) + np.array(f_d)) * dt
#            Psi.append(u_n_1)
#
#        return Psi
#
##        w = [self.__omega0]
##        for i in w:
##            self.__omega = i
##            Psi0 = self.__alpha0, self.__beta0
##            t_max = 20 / self.__omega1
##            self.__time = np.arange(0, t_max, t_max/100000)
##            #solution = sp.solve_ivp(self.schrodinger, [0, t_max], Psi0, t_eval=self.__time)
##            solution = odeintz(self.schrodinger, Psi0, self.__time)
##            return solution
##            data = solution
##            plt.plot(QBit.omega(), data.y[1]**2, label='beta')
#
#        #plt.plot(QBit.omega(), data.y[0]**2, label='alpha')





