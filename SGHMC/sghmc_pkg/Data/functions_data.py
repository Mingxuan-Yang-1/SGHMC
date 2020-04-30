# Real Data
import numpy as np
import numba

# SGHMC
def sghmc_resample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo with 
    resampling procedure when a data set is given.
    """
    du_hat_func = numba.njit(du_hat_func)
    formula = numba.njit(formula)
    @numba.njit
    def jit_du(beta, X, y, m):
        return du_hat_func(beta, X, y, m)
    
    @numba.njit
    def jit_formula(x):
        return formula(x)
    
    n, p = X.shape
    assert n % m == 0
    theta = np.zeros((nt, p))
    theta[0] = theta_init
    r = np.random.multivariate_normal(np.zeros(p), M, nt)
    Minv = np.linalg.inv(M)
    noise = 2*(C - B_hat)
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter1 = epsilon0*Minv
        counter2 = C@counter1
        s2 = noise*epsilon0
        theta0, r0 = theta[t], r[t]
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + counter1@r0
            r0 = r0 - epsilon0*jit_du(theta0, datas[i][:, :-1], datas[i][:, -1], m) - counter2@r0 + np.random.multivariate_normal(np.zeros(p), s2)
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
    return [theta, r]

def sghmc_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo without 
    resampling procedure when a data set is given.
    """
    du_hat_func = numba.njit(du_hat_func)
    formula = numba.njit(formula)
    @numba.njit
    def jit_du(beta, X, y, m):
        return du_hat_func(beta, X, y, m)
    
    @numba.njit
    def jit_formula(x):
        return formula(x)
    
    n, p = X.shape
    assert n % m == 0
    theta = np.zeros((nt, p))
    r = np.zeros((nt, p))
    theta[0] = theta_init
    r[0] = r_init
    Minv = np.linalg.inv(M)
    noise = 2*(C - B_hat)
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter1 = epsilon0*Minv
        counter2 = C@counter1
        s2 = noise*epsilon0
        theta0, r0 = theta[t], r[t]
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + counter1@r0
            r0 = r0 - epsilon0*jit_du(theta0, datas[i][:, :-1], datas[i][:, -1], m) - counter2@r0 + np.random.multivariate_normal(np.zeros(p), s2)
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
        r[t+1] = r0
    return [theta, r]

def sghmc_summarize_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo under
    different conditions when a data set is given.
    The given data set should contain at least two predictors.
    formula: a function of iteration index t.
    """
    assert X.shape[0] == y.shape[0]
    assert isinstance(epsilon, (int, float)) and epsilon >0, 'epsilon should be a positive number'
    assert isinstance(m, int) and m>0, 'm should be a positive integer'
    assert isinstance(nt, int) and nt>0, 'nt should be a positive integer'
    assert isinstance(theta_init, (int,float,np.ndarray)), 'theta_init should either be a number or a numpy array'
    assert isinstance(r_init, (int,float,np.ndarray)) or r_init is None, 'r_init should either be None, or a number/numpy array'
    assert isinstance(resample, bool), 'resample should be a boolean'
    assert M.shape == B_hat.shape == C.shape, 'M, B_hat and C should have the same shape'
    
    if resample:
        # resampling
        return sghmc_resample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula)
    else:
        # no resampling
        assert theta_init.shape == r_init.shape, 'theta and r should have the same shape'
        return sghmc_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula)

# SGLD
def sgld_mh_resample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction with resampling procedure when a data set
    is given.
    """
    n = X.shape[0]
    theta = [theta_init]
    r = []
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1], datas[i][:, :-1], datas[i][:, -1], m) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        pp = np.exp(H2 - H1)
        if u < min(1,pp):
            theta.append(theta0)
    return [theta[:-1], r]
            
def sgld_nomh_resample_data(X, y, du_hat_func, epsilon, nt, m, M, B, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction with resampling procedure when a data set is
    given.
    """
    n = X.shape[0]
    theta = [theta_init]
    r = []
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def sgld_mh_noresample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction without resampling procedure when a data
    set is given.
    """
    n = X.shape[0]
    theta = [theta_init]
    r = [r_init]
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1], datas[i][:, :-1], datas[i][:, -1], m) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        pp = np.exp(H2 - H1)
        if u < min(1,pp):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def sgld_nomh_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction without resampling procedure when a data set
    is given.
    """
    n = X.shape[0]
    theta = [theta_init]
    r = [r_init]
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def sgld_summarize_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula, MH = True, resample = True):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics under different 
    conditions when a data set is given.
    The given data set should contain at least two predictors.
    formula: a function of iteration index t.
    """
    if resample:
        # resampling
        if MH:
            # Metropolis-Hastings correction
            return sgld_mh_resample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, formula)
        else:
            # No Metropolis-Hastings correction
            return sgld_nomh_resample_data(X, y, du_hat_func, epsilon, nt, m, M, B, theta_init, formula)
    else:
        # no resampling
        if MH:
            # Metropolis-Hastings correction
            return sgld_mh_noresample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula)
        else:
            # No Metropolis-Hastings correction
            return sgld_nomh_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula)

# SGD with momentum
def sgdwm_mh_resample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Descent with momentum with 
    Metropolis-Hastings correction with resampling procedure when a data set is 
    give.
    """
    n, p = X.shape
    theta = [theta_init]
    v = []
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        v.append(np.random.multivariate_normal(np.zeros(p), M))
        theta0, v0 = theta[-1], v[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + v0
            v0 = v0 - epsilon**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon*np.linalg.inv(M)@C@v0 + np.random.multivariate_normal(np.zeros(p), 2*epsilon0**3*np.linalg.inv(M)@(C - B_hat)@np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + 1/2*v0.T@np.linalg.inv(M)@v0
        H2 = u_hat_func(theta[-1], datas[i][:, :-1], datas[i][:, -1], m) + 1/2*v[-1].T@np.linalg.inv(M)@v[-1]
        pp = np.exp(H2 - H1)
        if u < min(1,pp):
            theta.append(theta0)
    return [theta[:-1], v]

def sgdwm_nomh_resample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Descent with momentum without 
    Metropolis-Hastings correction with resampling procedure when a data set is given.
    """
    n, p = X.shape
    theta = np.zeros((nt, p))
    theta[0] = theta_init
    v = np.random.multivariate_normal(np.zeros(M.shape[0]), M, nt)
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, v0 = theta[-1], v[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + v0
            v0 = v0 - epsilon**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon*np.linalg.inv(M)@C@v0 + np.random.multivariate_normal(np.zeros(p), 2*epsilon0**3*np.linalg.inv(M)@(C - B_hat)@np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
    return [theta, v]

def sgdwm_mh_noresample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula):
    """
    This is a function to realize Stochastic Gradient Descent with momentum with
    Metropolis-Hastings correction without resampling procedure when a data set is
    given.
    """
    n, p = X.shape
    theta = [theta_init]
    v = [v_init]
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, v0 = theta[-1], v[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + v0
            v0 = v0 - epsilon**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon*np.linalg.inv(M)@C@v0 + np.random.multivariate_normal(np.zeros(p), 2*epsilon0**3*np.linalg.inv(M)@(C - B_hat)@np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) + 1/2*v0.T@np.linalg.inv(M)@v0
        H2 = u_hat_func(theta[-1], datas[i][:, :-1], datas[i][:, -1], m) + 1/2*v[-1].T@np.linalg.inv(M)@v[-1]
        pp = np.exp(H2 - H1)
        if u < min(1,pp):
            theta.append(theta0)
            v.append(v0)
    return [theta, v]

def sgdwm_nomh_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula):
    """
    This is a function to realize Stochastic Gradient Descent with momentum without 
    Metropolis-Hastings correction without resampling procedure when a data set is 
    given.
    """
    n, p = X.shape
    theta = [theta_init]
    v = [v_init]
    data = np.hstack((X, y.reshape(-1,1)))
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, v0 = theta[-1], v[-1]
        assert n % m == 0
        np.random.shuffle(data)
        datas = np.split(data, m)
        for i in range(m):
            theta0 = theta0 + v0
            v0 = v0 - epsilon**2*np.linalg.inv(M)@du_hat_func(theta0, datas[i][:, :-1], datas[i][:, -1], m) - epsilon*np.linalg.inv(M)@C@v0 + np.random.multivariate_normal(np.zeros(p), 2*epsilon0**3*np.linalg.inv(M)@(C - B_hat)@np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta.append(theta0)
        v.append(v0)
    return [theta, v]

def sgdwm_summarize_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula, MH = True, resample = True):
    """
    This is a function to realize Stochastic Gradient Descent with momentum under 
    different conditions when a data set is given.
    The given data set should contain at least two predictors.
    formula: a function of iteration index t.
    """
    if resample:
        # resampling
        if MH:
            # Metropolis-Hastings correction
            return sgdwm_mh_resample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula)
        else:
            # No Metropolis-Hastings correction
            return sgdwm_nomh_resample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula)
    else:
        # no resampling
        if MH:
            # Metropolis-Hastings correction
            return sgdwm_mh_noresample_data(X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula)
        else:
            # No Metropolis-Hastings correction
            return sgdwm_nomh_noresample_data(X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula)