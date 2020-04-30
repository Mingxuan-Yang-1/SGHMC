from sghmc_pkg.algorithms_data import SGHMC_data, SGLD_data, SGDwm_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# For logistic regression
def du_hat_func(beta, X, y, m):
    z = y - 1/(1 + np.exp(-np.dot(X, beta)))
    d1 = X.T @ z
    d2 = - np.dot(np.linalg.inv(sigma), beta)
    return(- m*d1 - d2)

# test error
def test_error(X, y, theta):
    y_pred = (1/(1 + np.exp(- X @ np.array(theta).T)) > 0.5).astype(int)
    z = ((y[:,None] - y_pred)**2).sum(axis = 0)/len(y)
    return z

# training data
data_train = pd.read_csv('logis.csv')
data_train.drop('Unnamed: 0', axis = 1, inplace = True)
data_train.date = pd.to_datetime(data_train.date)
data_train.date = data_train.date.map(dt.datetime.toordinal)
X = np.array(data_train.loc[:5699, :'HumidityRatio'])
X = preprocessing.scale(X)/10
y = np.array(data_train.loc[:5699, 'Occupancy'])

# testing data
data_test = pd.read_csv('logis_test.csv')
data_test.drop('Unnamed: 0',axis = 1, inplace = True)
data_test.date = pd.to_datetime(data_test.date)
data_test.date = data_test.date.map(dt.datetime.toordinal)
X_test = np.array(data_test.loc[:,:'HumidityRatio'])
X_test = preprocessing.scale(X_test)/10
y_test = np.array(data_test.loc[:, 'Occupancy'])

# SGHMC
n, p = X.shape
sigma = np.eye(p)
epsilon = 0.001
nt = 1000
m = 10
M = C = B_hat = np.eye(p)
theta_init = r_init = np.ones(p)
formula = lambda x: 0
theta_sghmc = SGHMC_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample = True).theta()

# SGLD
B = np.eye(p)
epsilon = 0.001
theta_sgld = SGLD_data(X, y, None, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula, MH = False, resample = False).theta()

# SGD with momentum
epsilon = 0.001
m = 5
v_init = r_init
theta_sgdwm = SGDwm_data(X, y, None, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula, MH = False, resample = False).theta()

# plot
xs = np.arange(50,500,30)
plt.plot(xs, test_error(X_test, y_test, theta_sghmc[50:500][::30]), label = 'SGHMC')
plt.plot(xs, test_error(X_test, y_test, theta_sgld[50:500][::30]), label = 'SGLD')
plt.plot(xs, test_error(X_test, y_test, theta_sgdwm[50:500][::30]), label = 'SGD with momentum')
plt.legend(loc = 'upper right')
plt.show()