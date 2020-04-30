from sghmc_pkg.Data.functions_data import sghmc_summarize_data, sgld_summarize_data, sgdwm_summarize_data

class SGHMC_data:

    def __init__(self, X, y, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, r_init, formula, resample = True):
        self.X = X
        self.y = y
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
        return sghmc_summarize_data(self.X, self.y, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.r_init, self.formula, self.resample)[0]

    def r(self):
        return sghmc_summarize_data(self.X, self.y, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.r_init, self.formula, self.resample)[1]

class SGLD_data:

    def __init__(self, X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, B, theta_init, r_init, formula, MH = True, resample = True):
        self.X = X
        self.y = y
        self.u_hat_func = u_hat_func
        self.du_hat_func = du_hat_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.B = B
        self.theta_init = theta_init
        self.r_init = r_init
        self.formula = formula
        self.MH = MH
        self.resample = resample

    def theta(self):
        return sgld_summarize_data(self.X, self.y, self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.B, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[0]

    def r(self):
        return sgld_summarize_data(self.X, self.y, self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.B, self.theta_init, self.r_init, self.formula, self.MH, self.resample)[1]

class SGDwm_data:

    def __init__(self, X, y, u_hat_func, du_hat_func, epsilon, nt, m, M, C, B_hat, theta_init, v_init, formula, MH = True, resample = True):
        self.X = X
        self.y = y
        self.u_hat_func = u_hat_func
        self.du_hat_func = du_hat_func
        self.epsilon = epsilon
        self.nt = nt
        self.m = m
        self.M = M
        self.C = C
        self.B_hat = B_hat
        self.theta_init = theta_init
        self.v_init = v_init
        self.formula = formula
        self.MH = MH
        self.resample = resample

    def theta(self):
        return sgdwm_summarize_data(self.X, self.y, self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.v_init, self.formula, self.MH, self.resample)[0]

    def r(self):
        return sgdwm_summarize_data(self.X, self.y, self.u_hat_func, self.du_hat_func, self.epsilon, self.nt, self.m, self.M, self.C, self.B_hat, self.theta_init, self.v_init, self.formula, self.MH, self.resample)[1]