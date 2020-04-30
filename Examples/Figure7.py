from sghmc_pkg.algorithms_sim import HMC_sim, SGHMC_NAIVE_sim, SGHMC_sim, SGLD_sim
import matplotlib.pyplot as plt
import numpy as np

# SGHMC
sigma = np.array([1, 0.9, 0.9, 1]).reshape(2, 2)
u_func = lambda x: 1/2 * x.T @ np.linalg.inv(sigma) @ x
du_hat_func = lambda x: np.linalg.inv(sigma) @ x + np.random.multivariate_normal(np.array([0,0]), np.eye(2))
epsilon = 0.1
nt = 100
M = np.eye(2)
V = np.eye(2)
B_hat = C = 1/2*epsilon*V
theta_init = np.array([0,0])
r_init = np.random.multivariate_normal(np.array([0,0]), M)
formula = lambda x: 0
theta_sghmc = SGHMC_sim(du_hat_func, epsilon, nt, 1, M, C, B_hat, theta_init, r_init, formula, resample = False).theta()

# SGLD
theta_sgld = SGLD_sim(None, du_hat_func, epsilon, nt, 1, M, V, theta_init, r_init, formula, MH = False, resample = False).theta()

# plot
x = np.linspace(-2, 3, 100)
y = np.linspace(-2, 3, 100)
xx, yy = np.meshgrid(x, y)
a, b, c = np.linalg.inv(sigma)[0,0], np.linalg.inv(sigma)[1,1], np.linalg.inv(sigma)[1,0]
z = np.exp(-1/2*(a*xx**2 + b*yy**2 + 2*c*xx*yy))
plt.subplots(figsize=(4,4))
plt.contour(x, y, z, levels = 6)
plt.scatter(np.array(theta_sghmc).T[0][::2], np.array(theta_sghmc).T[1][::2], s = 10, color = 'red', marker = 'o', label = 'SGHMC')
plt.scatter(np.array(theta_sgld).T[0][::2], np.array(theta_sgld).T[1][::2], color = 'blue', marker = 'x', label = 'SGLD')
plt.legend(loc="upper right")
plt.show()