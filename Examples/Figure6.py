from sghmc_pkg.algorithms_sim import HMC_sim, SGHMC_NAIVE_sim, SGHMC_sim, SGLD_sim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import quad

# true distribution
p_func = lambda x: np.exp(2*x**2 - x**4)
xs = np.linspace(-2, 2, 200)
ys = p_func(xs)/quad(p_func, float('-Inf'), float('Inf'))[0]

# standard HMC
u_func = lambda x: -2*x**2 + x**4
du_func = lambda x: -4*x + 4*x**3
epsilon = 0.1
nt = 10000
m = 50
M = 1
theta_init = 0
theta_hmc_mh = HMC_sim(u_func, du_func, epsilon, nt, m, M, theta_init, None, MH = True, resample = True).theta()
theta_hmc_nomh = HMC_sim(u_func, du_func, epsilon, nt, m, M, theta_init, None, MH = False, resample = True).theta()

# Naive stochastic gradient HMC
V = 4
u_hat_func = u_func
formula = lambda x: 0
du_hat_func = lambda x:  -4*x + 4*x**3 + np.random.normal(0, np.sqrt(V))
theta_naive_mh = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, None, formula, MH = True, resample = True).theta()
theta_naive_nomh = SGHMC_NAIVE_sim(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, None, formula, MH = False, resample = True).theta()

# SGHMC
B_hat = C = 1/2*epsilon*V
theta_sghmc = SGHMC_sim(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, None, formula, resample = True).theta()

# plot
plt.subplots(figsize=(7, 4))
plt.plot(xs, ys, color = 'red', label = 'True Distribution')
sns.distplot(theta_hmc_mh, hist = False, color = 'orange', label = 'Standard HMC (with MH)')
sns.distplot(theta_hmc_nomh, hist = False, color = 'green', label = 'Standard HMC (no MH)')
sns.distplot(theta_naive_mh, hist = False, color = 'black', label = 'Naive SGHMC (with MH)')
sns.distplot(theta_naive_nomh, hist = False, color = 'pink', label = 'Naive SGHMC (no MH)')
sns.distplot(theta_sghmc, hist = False, color = 'blue', label = 'SGHMC')
plt.axis([-2, 2, 0, 1,])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()