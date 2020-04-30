from sghmc_pkg.algorithms_data import SGHMC_data, SGLD_data, SGDwm_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

# For linear regression
def du_hat_func(beta, X, y, m):
    # prior: beta ~ N(np.zeros(p), I/phi)
    d1 = X.T @ X @ beta - X.T @ y
    d2 = beta
    return(m*d1 + d2)

# test error
def test_error(X, y, theta):
    y_pred = X @ np.array(theta).T
    z = ((y[:,None] - y_pred)**2).sum(axis = 0)/len(y)
    return(z)

# data
# https://www.kaggle.com/uciml/adult-census-income
data = pd.read_csv('insurance.csv')
data.sex = data.sex.factorize()[0]
data.smoker = data.smoker.factorize()[0]
data.region = data.region.factorize()[0]
data = preprocessing.scale(data)
X = data[:1000, :-1]
X_test = data[1000:, :-1]
y = data[:1000, -1]
y_test = data[1000:, -1]

# SGHMC
n, p = X.shape
epsilon = 0.005
nt = 1000
m = 5
M = C = B_hat = np.eye(p)
theta_init = r_init = np.ones(p)
formula = lambda x: 0
theta_sghmc = SGHMC_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample = False).theta()

# SGLD
B = np.eye(p)
epsilon = 0.0009
theta_sgld = SGLD_data(X, y, None, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula, MH = False, resample = False).theta()

# SGD with momentum
v_init = epsilon*np.linalg.inv(M)@r_init
theta_sgdwm = SGDwm_data(X, y, None, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula, MH = False, resample = False).theta()

# plot
xs = np.arange(20,500,50)
plt.plot(xs, test_error(X_test, y_test, theta_sghmc[20:500][::50]), label = 'SGHMC')
plt.plot(xs, test_error(X_test, y_test, theta_sgld[20:500][::50]), label = 'SGLD')
plt.plot(xs, test_error(X_test, y_test, theta_sgdwm[20:500][::50]), label = 'SGD with momentum')
plt.legend(loc = 'upper right')
plt.show()