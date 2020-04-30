from sghmc_pkg.Simulation.functions_sim import hmc_summarize, sghmc_naive_summarize, sghmc_summarize, sgld_summarize

class HMC_sim:

    def __init__(self, u_func, du_func, epsilon, nt, m, M, theta_init, r_init, MH = True, resample = True):
        self.u_func = u_func
        self.du_func = du_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.theta_init = theta_init
        self.r_init = r_init
        self.MH = MH
        self.resample = resample

    def theta(self):
        return hmc_summarize(self.u_func, self.du_func, self.epsilon, self.nt, self.m, self.M, self.theta_init, self.r_init, self.MH, self.resample)[0]

    def r(self):
        return hmc_summarize(self.u_func, self.du_func, self.epsilon, self.nt, self.m, self.M, self.theta_init, self.r_init, self.MH, self.resample)[1]

class SGHMC_NAIVE_sim:

    def __init__(self, u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula, MH = True, resample = True):
        self.u_hat_func = u_hat_func
        self.du_hat_func = du_hat_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.V = V
        self.theta_init = theta_init
        self.r_init = r_init
        self.formula = formula
        self.MH = MH
        self.resample = resample

    def theta(self):
        return sghmc_naive_summarize(self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.V, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[0]

    def r(self):
        return sghmc_naive_summarize(self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.V, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[1]

class SGHMC_sim:

    def __init__(self, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample = True):
        self.du_hat_func = du_hat_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.C = C
        self.B_hat = B_hat
        self.theta_init = theta_init
        self.r_init = r_init
        self.formula = formula
        self.resample = resample

    def theta(self):
        return sghmc_summarize(self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.r_init, self.formula, self.resample)[0]

    def r(self):
        return sghmc_summarize(self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.r_init, self.formula, self.resample)[1]

class SGLD_sim:

    def __init__(self, u_hat_func, du_hat_func, epsilon, nt, m, M, V, theta_init, r_init, formula, MH = True, resample = True):
        self.u_hat_func = u_hat_func
        self.du_hat_func = du_hat_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.V = V
        self.theta_init = theta_init
        self.r_init = r_init
        self.formula = formula
        self.MH = MH
        self.resample = resample

    def theta(self):
        return sgld_summarize(self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.V, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[0]

    def r(self):
        return sgld_summarize(self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.V, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[1]