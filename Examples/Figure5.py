from sghmc_pkg.algorithms_sim import HMC_sim, SGHMC_NAIVE_sim, SGHMC_sim, SGLD_sim
import matplotlib.pyplot as plt
import numpy as np

# Noisy Hamiltonian dynamics
u_func = lambda x: 1/2*x**2
du_func = lambda x: x
u_hat_func = lambda x: 1/2*x**2
du_hat_func = lambda x: x + np.random.normal(0, np.sqrt(V))
epsilon = 0.1
nt = 1000
V = 4
M = 1
theta_init = 0
r_init = 1
formula = lambda x: 0
theta_noisy = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, 1, M, V, theta_init, r_init, formula, MH = False, resample = False).theta()[::5]
r_noisy = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, 1, M, V, theta_init, r_init, formula, MH = False, resample = False).r()[::5]

# Noisy Hamiltonian dynamics (resample r each 50 steps)
nt = 300
m = 50
theta_noisy_resample = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, None, formula, MH = False, resample = True).theta()
r_noisy_resample = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, None, formula, MH = False, resample = True).r()

# Noisy Hamiltonian dynamics with friction
B_hat = C = 1/2*epsilon*V
theta_sghmc_2 = SGHMC_sim(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, None, formula, resample = True).theta()
r_sghmc_2 = SGHMC_sim(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, None, formula, resample = True).r()

# Hamiltonian dynamics
nt = 1000
theta_hmc_2 = HMC_sim(u_func, du_func, epsilon, nt, 1, M, theta_init, r_init, MH = False, resample = False).theta()
r_hmc_2 = HMC_sim(u_func, du_func, epsilon, nt, 1, M, theta_init, r_init, MH = False, resample = False).r()

# plot
plt.scatter(theta_noisy, r_noisy, color = 'red', marker = 'o', label = 'Noisy Hamiltonian dynamics')
plt.scatter(theta_noisy_resample, r_noisy_resample, color = 'black', marker = '+', label = 'Noisy Hamiltonian dynamics(resample r each 50 steps)')
plt.scatter(theta_sghmc_2, r_sghmc_2, color = 'green', marker = '*', label = 'Noisy Hamiltonian dynamics with friction')
plt.scatter(theta_hmc_2, r_hmc_2, color = 'blue', marker = 'x', label = 'Hamiltonian dynamics')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis([-8, 8, -8, 8,])
plt.show()