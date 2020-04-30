# Simulation
import numpy as np
import numba
from numba import jit

# standard HMC
def hmc_mh_resample_uni(u_func, du_func, epsilon, nt, m, M, theta_init):
    """
    This is a function to realize Hamiltonian Monte Carlo with Metropolis-Hastings 
    correction in unidimensional cases with resampling procedure.
    """
    theta = [theta_init]
    r = []
    for t in range(nt):
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*1/M*r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_func(theta0) + 1/2*r0**2*1/M
        H2 = u_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]
    
def hmc_nomh_resample_uni(du_func, epsilon, nt, m, M, theta_init):
    """
    This is a function to realize Hamiltonian Monte Carlo without Metropolis-Hastings 
    correction in unidimensional cases with resampling procedure.
    """
    theta = [theta_init]
    r = []
    for t in range(nt):
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*1/M*r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def hmc_mh_resample_multi(u_func, du_func, epsilon, nt, m, M, theta_init):
    """
    This is a function to realize Hamiltonian Monte Carlo with Metropolis-Hastings 
    correction in multidimensional cases with resampling procedure.
    """
    theta = [theta_init]
    r = []
    for t in range(nt):
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*np.linalg.inv(M)@r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]

def hmc_nomh_resample_multi(du_func, epsilon, nt, m, M, theta_init):
    """
    This is a function to realize Hamiltonian Monte Carlo without Metropolis-Hastings 
    correction in unidimensional cases with resampling procedure.
    """
    theta = [theta_init]
    r = []
    for t in range(nt):
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*np.linalg.inv(M)@r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def hmc_mh_noresample_uni(u_func, du_func, epsilon, nt, m, M, theta_init, r_init):
    """
    This is a function to realize Hamiltonian Monte Carlo with Metropolis-Hastings 
    correction in unidimensional cases without resampling procedure.
    """
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*1/M*r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_func(theta0) + 1/2*r0**2*1/M
        H2 = u_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]
    
def hmc_nomh_noresample_uni(du_func, epsilon, nt, m, M, theta_init, r_init):
    """
    This is a function to realize Hamiltonian Monte Carlo without Metropolis-Hastings 
    correction in unidimensional cases without resampling procedure.
    """
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*1/M*r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def hmc_mh_noresample_multi(u_func, du_func, epsilon, nt, m, M, theta_init, r_init):
    """
    This is a function to realize Hamiltonian Monte Carlo with Metropolis-Hastings 
    correction in multidimensional cases without resampling procedure.
    """
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*np.linalg.inv(M)@r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def hmc_nomh_noresample_multi(du_func, epsilon, nt, m, M, theta_init, r_init):
    """
    This is a function to realize Hamiltonian Monte Carlo without Metropolis-Hastings 
    correction in multidimensional cases without resampling procedure.
    """
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        theta0, r0 = theta[-1], r[-1]
        r0 = r0 - epsilon/2*du_func(theta0)
        for i in range(m):
            theta0 = theta0 + epsilon*np.linalg.inv(M)@r0
            r0 = r0 - epsilon*du_func(theta0)
        r0 = r0 - epsilon/2*du_func(theta0)
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def hmc_summarize(u_func, du_func, epsilon, nt, m, M, theta_init, r_init, MH = True, resample = True):
    """
    This is a function to realize Hamiltonian Monte Carlo under different conditions.
    If theta_init is unidimensional, it needs to be a numeric number.
    If theta_init is multidimensional, it needs to be an array.
    formula: a function of iteration index t.
    """
    if isinstance(theta_init, np.ndarray):
        # multidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return hmc_mh_resample_multi(u_func, du_func, epsilon, nt, m, M, theta_init)
            else:
                # No Metropolis-Hastings correction
                return hmc_nomh_resample_multi(du_func, epsilon, nt, m, M, theta_init)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return hmc_mh_noresample_multi(u_func, du_func, epsilon, nt, m, M, theta_init, r_init)
            else:
                # No Metropolis-Hastings correction
                return hmc_nomh_noresample_multi(du_func, epsilon, nt, m, M, theta_init, r_init)
            
    else:
        # unidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return hmc_mh_resample_uni(u_func, du_func, epsilon, nt, m, M, theta_init)
            else:
                # No Metropolis-Hastings correction
                return hmc_nomh_resample_uni(du_func, epsilon, nt, m, M, theta_init)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return hmc_mh_noresample_uni(u_func, du_func, epsilon, nt, m, M, theta_init, r_init)
            else:
                # No Metropolis-Hastings correction
                return hmc_nomh_noresample_uni(du_func, epsilon, nt, m, M, theta_init, r_init)

# Naive SGHMC
def sghmc_naive_mh_resample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    with Metropolis-Hastings correction in unidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*1/M*r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.normal(0, np.sqrt(2*B*epsilon0))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0**2*1/M
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]

def sghmc_naive_nomh_resample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    without Metropolis-Hastings correction in unidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*1/M*r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.normal(0, np.sqrt(2*B*epsilon0))
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def sghmc_naive_mh_resample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    with Metropolis-Hastings correction in multidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*np.linalg.inv(M)@r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0*B)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]
            
def sghmc_naive_nomh_resample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    without Metropolis-Hastings correction in multidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*np.linalg.inv(M)@r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0*B)
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def sghmc_naive_mh_noresample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    with Metropolis-Hastings correction in unidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*1/M*r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.normal(0, np.sqrt(2*B*epsilon0))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0**2*1/M
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def sghmc_naive_nomh_noresample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    without Metropolis-Hastings correction in unidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*1/M*r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.normal(0, np.sqrt(2*B*epsilon0))
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def sghmc_naive_mh_noresample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    with Metropolis-Hastings correction in multidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*np.linalg.inv(M)@r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0*B)
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def sghmc_naive_nomh_noresample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    without Metropolis-Hastings correction in multidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            theta0 = theta0 + epsilon0*1/M*r0
            r0 = r0 - epsilon0*du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0*B)
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def sghmc_naive_summarize(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula, MH = True, resample = True):
    """
    This is a function to realize Naive Stochastic Gradient Hamiltonian Monte Carlo 
    under different conditions.
    If theta_init is unidimensional, it needs to be a numeric number.
    If theta_init is multidimensional, it needs to be an array.
    formula: a function of iteration index t.
    """
    if isinstance(theta_init, np.ndarray):
        # multidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return sghmc_naive_mh_resample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sghmc_naive_nomh_resample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return sghmc_naive_mh_noresample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sghmc_naive_nomh_noresample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)
        
    else:
        # unidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return sghmc_naive_mh_resample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sghmc_naive_nomh_resample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return sghmc_naive_mh_noresample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sghmc_naive_nomh_noresample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)

# SGHMC
def sghmc_resample_uni(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo in 
    unidimensional cases with resampling procedure.
    """
    du_hat_func = numba.njit(du_hat_func)
    formula = numba.njit(formula)
    @numba.njit
    def jit_du(x):
        return du_hat_func(x)
    @numba.njit
    def jit_formula(x):
        return formula(x)
    theta = np.zeros(nt)
    theta[0] = theta_init
    prac = 2*(C - B_hat)
    r = np.random.normal(0, np.sqrt(M), size = nt)
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter = epsilon0*C/M
        sd = np.sqrt(prac*epsilon0)
        theta0, r0 = theta[t], r[t]
        noise = np.random.normal(0,sd,size = m)
        for i in range(m):
            theta0 = theta0 + epsilon0/M*r0
            r0 = r0 - epsilon0*jit_du(theta0) - counter*r0 + noise[i]
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
    return [theta, r]

def sghmc_resample_multi(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo in 
    multidimensional cases with resampling procedure.
    """
    formula = numba.njit(formula)
    @jit(forceobj = True)
    def jit_du(x):
        return du_hat_func(x)
    @numba.njit
    def jit_formula(x):
        return formula(x)
    p = theta_init.shape[0]
    theta = np.zeros((nt,p))
    r = np.random.multivariate_normal(np.zeros(p), M, size = nt)
    theta[0,:] = theta_init
    Min = np.linalg.inv(M)
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter = epsilon0*C@Min
        cov = 2*epsilon0*(C - B_hat)
        noise = np.random.multivariate_normal(np.zeros(p), cov, size = m)
        theta0, r0 = theta[t,:], r[t,:]
        for i in range(m):
            theta0 = theta0 + epsilon0*Min@r0
            r0 = r0 - epsilon0*jit_du(theta0) - counter@r0 + noise[i,:]
        # No Metropolis-Hastings correction
        theta[t+1,:] = theta0
    return [theta, r]

def sghmc_noresample_uni(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo in 
    unidimensional cases without resampling procedure.
    """
    du_hat_func = numba.njit(du_hat_func)
    formula = numba.njit(formula)
    @numba.njit
    def jit_du(x):
        return du_hat_func(x)
    @numba.njit
    def jit_formula(x):
        return formula(x)
    theta = np.zeros(nt)
    r = np.zeros(nt)
    theta[0] = theta_init
    r[0] = r_init
    prac = 2*(C - B_hat)
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter = epsilon0*C*1/M
        sd = np.sqrt(prac*epsilon0)
        noise = np.random.normal(0, sd, size = m)
        theta0, r0 = theta[t], r[t]
        for i in range(m):
            theta0 = theta0 + epsilon0/M*r0
            r0 = r0 - epsilon0*jit_du(theta0) - counter*r0 + noise[i]
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
        r[t+1] = r0
    return [theta, r]

def sghmc_noresample_multi(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo in 
    multidimensional cases without resampling procedure.
    """
    formula = numba.njit(formula) 
    @jit(forceobj = True)
    def jit_du(x):
        return du_hat_func(x)
    @numba.njit
    def jit_formula(x):
        return formula(x)
    p = theta_init.shape[0]
    theta = np.zeros((nt,p))
    r = np.zeros((nt,p))
    theta[0,:] = theta_init
    r[0,:] = r_init
    Min = np.linalg.inv(M)
    for t in range(nt-1):
        epsilon0 = max(epsilon, jit_formula(t))
        counter = epsilon0*C@Min
        cov = 2*epsilon0*(C - B_hat)
        theta0, r0 = theta[t], r[t]
        noise = np.random.multivariate_normal(np.zeros(p), cov, size = m)
        for i in range(m):
            theta0 = theta0 + epsilon0*Min@r0
            r0 = r0 - epsilon0*jit_du(theta0) - counter@r0 + noise[i,:]
        # No Metropolis-Hastings correction
        theta[t+1] = theta0
        r[t+1] = r0
    return [theta, r]

def sghmc_summarize(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample):
    """
    This is a function to realize Stochastic Gradient Hamiltonian Monte Carlo under
    different conditions.
    If theta_init is unidimensional, it needs to be a numeric number.
    If theta_init is multidimensional, it needs to be an array.
    formula: a function of iteration index t.
    """
    assert isinstance(epsilon, (int, float)) and epsilon >0, 'epsilon should be a positive number'
    assert isinstance(m, int) and m>0, 'm should be a positive integer'
    assert isinstance(nt, int) and nt>0, 'nt should be a positive integer'
    assert isinstance(theta_init, (int,float,np.ndarray)), 'theta_init should either be a number or a numpy array'
    assert isinstance(r_init, (int,float,np.ndarray)) or r_init is None, 'r_init should either be None, or a number/numpy array'
    assert isinstance(resample, bool), 'resample should be a boolean'
    
    if isinstance(theta_init, np.ndarray):
        # multidimensional cases
        assert M.shape == B_hat.shape == C.shape, 'M, B_hat and C should have the same shape'
        if resample:
            # resampling
            return sghmc_resample_multi(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula)
        else:
            # no resampling
            assert theta_init.shape == r_init.shape, 'theta and r should have the same shape'
            return sghmc_noresample_multi(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula)
            
    else:
        # unidimensional cases
        assert M > 0 and B_hat >= 0, 'M should be greater than 0, and B_hat should be no less than 0'
        assert C-B_hat >= 0, 'C should be no less than B_hat'
        if resample:
            # resampling
            return sghmc_resample_uni(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, formula)
        else:
            # no resampling
            return sghmc_noresample_uni(du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula)

# SGLD
def sgld_mh_resample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction in unidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B*1/M*r0 + np.random.normal(0, np.sqrt(2*B*epsilon0))
            theta0 = theta0 - 1/M*du_hat_func(theta0)*epsilon0**2 + np.random.normal(0, np.sqrt(2*1/M*epsilon0**2))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0**2*1/M
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]
            
def sgld_nomh_resample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction in unidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.normal(0, np.sqrt(M)))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B*1/M*r0 + np.random.normal(0, np.sqrt(2*B*epsilon0))
            theta0 = theta0 - 1/M*du_hat_func(theta0)*epsilon0**2 + np.random.normal(0, np.sqrt(2*1/M*epsilon0**2))
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def sgld_mh_resample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction in multidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
    return [theta[:-1], r]
            
def sgld_nomh_resample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction in multidimensional cases with resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = []
    for t in range(nt):
        epsilon0 = max(epsilon, formula(t))
        r.append(np.random.multivariate_normal(np.zeros(M.shape[0]), M))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta.append(theta0)
    return [theta[:-1], r]

def sgld_mh_noresample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction in unidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B*1/M*r0 + np.random.normal(0, np.sqrt(2*B*epsilon0))
            theta0 = theta0 - 1/M*du_hat_func(theta0)*epsilon0**2 + np.random.normal(0, np.sqrt(2*1/M*epsilon0**2))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0**2*1/M
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1]**2*1/M
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def sgld_nomh_noresample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction in unidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B*1/M*r0 + np.random.normal(0, np.sqrt(2*B*epsilon0))
            theta0 = theta0 - 1/M*du_hat_func(theta0)*epsilon0**2 + np.random.normal(0, np.sqrt(2*1/M*epsilon0**2))
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def sgld_mh_noresample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics with 
    Metropolis-Hastings correction in multidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # Metropolis-Hastings correction
        u = np.random.uniform()
        H1 = u_hat_func(theta0) + 1/2*r0.T@np.linalg.inv(M)@r0
        H2 = u_hat_func(theta[-1]) + 1/2*r[-1].T@np.linalg.inv(M)@r[-1]
        p = np.exp(H2 - H1)
        if u < min(1,p):
            theta.append(theta0)
            r.append(r0)
    return [theta, r]

def sgld_nomh_noresample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics without 
    Metropolis-Hastings correction in multidimensional cases without resampling 
    procedure.
    """
    B = 1/2*epsilon*V
    theta = [theta_init]
    r = [r_init]
    for t in range(nt-1):
        epsilon0 = max(epsilon, formula(t))
        theta0, r0 = theta[-1], r[-1]
        for i in range(m):
            r0 = r0 - epsilon0*du_hat_func(theta0) - epsilon0*B@np.linalg.inv(M)@r0 + np.random.multivariate_normal(np.zeros(M.shape[0]), 2 * epsilon0 * B)
            theta0 = theta0 - epsilon0**2*np.linalg.inv(M)@du_hat_func(theta0) + np.random.multivariate_normal(np.zeros(M.shape[0]), 2*epsilon0**2*np.linalg.inv(M))
        # No Metropolis-Hastings correction
        theta.append(theta0)
        r.append(r0)
    return [theta, r]

def sgld_summarize(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula, MH = True, resample = True):
    """
    This is a function to realize Stochastic Gradient Langevin dynamics under different 
    conditions.
    If theta_init is unidimensional, it needs to be a numeric number.
    If theta_init is multidimensional, it needs to be an array.
    formula: a function of iteration index t.
    """
    if isinstance(theta_init, np.ndarray):
        # multidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return sgld_mh_resample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sgld_nomh_resample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return sgld_mh_noresample_multi(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sgld_nomh_noresample_multi(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)

    else:
        # unidimensional cases
        if resample:
            # resampling
            if MH:
                # Metropolis-Hastings correction
                return sgld_mh_resample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sgld_nomh_resample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, formula)
        else:
            # no resampling
            if MH:
                # Metropolis-Hastings correction
                return sgld_mh_noresample_uni(u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)
            else:
                # No Metropolis-Hastings correction
                return sgld_nomh_noresample_uni(du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula)