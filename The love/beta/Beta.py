import scipy.special as ss
import numpy as np


class beta_calculator():
    def __init__(self,raw_count,designmatrix,disp,variance_beta,size_factors,beta_init):
        self.K = np.array(raw_count)
        self.designmatrix = designmatrix
        self.alpha = disp
        self.s = size_factors
        self.init = beta_init
        self.res = None
        # self.normal = normalized_count
    ## Loglikelihood Functions
    def logNegativeBinomial(self , K , alpha , mu):
        return ss.loggamma(K +1/alpha)- ss.loggamma(1/alpha)  - K * np.log(mu + 1/alpha)- (1/alpha) * np.log(1+alpha*mu) 
      
    def fitBeta(self,start,end,minmu=0.5,tol=1e-8,maxit=100):
        y = self.K
        x = self.designmatrix
        nf = self.s.squeeze()
        alpha_hat = self.alpha
        beta_mat = self.init
        y_n,y_m = y.shape
        x_p = x.shape[1]
        Lambda = 1e-6*np.ones(x_p) / (np.log(2))**2
        beta_var_mat = np.zeros(beta_mat.shape)
        contrast_num = np.zeros((beta_mat.shape[0],1))
        contrast_denom = np.zeros((beta_mat.shape[0],1))
        hat_diagonals = np.zeros(y.shape)
        large = 30
        deviance = np.ones(y_n)
        for i in range(start,end):
            yrow = y[i]
            beta_hat = beta_mat[i]
            mu_hat = nf * np.exp(x@beta_hat)
            for j in range(y_m):
                mu_hat[j] = max(mu_hat[j],minmu)
            ridge = np.diag(Lambda)
            dev = 0.0
            dev_old = 0.0
            for t in range(maxit):
                w_vec = mu_hat / (1 + alpha_hat[i] * mu_hat)
                w_sqrt_vec = np.sqrt(w_vec)
                z = np.log(mu_hat / nf) + (yrow - mu_hat) / mu_hat
                beta_hat = np.linalg.solve((x.T)@(np.apply_along_axis(lambda x:x*w_vec,0,x)) + ridge,(x.T)@(z*w_vec))
                if (abs(beta_hat) > large).any():
                    break
                mu_hat = nf * np.exp(x@beta_hat)
                for j in range(y_m):
                    mu_hat[j] = max(mu_hat[j],minmu)
                dev = 0.0
                for j in range(y_m):
                    dev += -2 * self.logNegativeBinomial(yrow[j],alpha_hat[i],mu_hat[j])
                conv_test = abs(dev - dev_old) / (abs(dev) + .1)
                if np.isnan(conv_test):
                    break
                if t>0 and conv_test<tol:
                    break
                dev_old = dev
            deviance[i] = dev
            beta_mat[i] = beta_hat
            w_vec = mu_hat/(1.0 + alpha_hat[i] * mu_hat)
            w_sqrt_vec = np.sqrt(w_vec)
            hat_matrix_diag = np.zeros(x.shape[0])
            xw = np.apply_along_axis(lambda x:x*w_sqrt_vec,0,x)
            xtwxr_inv = np.linalg.inv((x.T) @ (np.apply_along_axis(lambda x:x*w_vec,0,x)) + ridge)
            for jp in range(y_m):
                for idx1 in range(x_p):
                    for idx2 in range(x_p):
                        hat_matrix_diag[jp] += xw[jp][idx1] * (xw[jp][idx2] * xtwxr_inv[idx2][idx1])
            hat_diagonals[i] = hat_matrix_diag
            sigma = np.linalg.inv(x.T @ (np.apply_along_axis(lambda x:x*w_vec,0,x)) + ridge) @ x.T @ (np.apply_along_axis(lambda x:x*w_vec,0,x)) @ np.linalg.inv(x.T @ (np.apply_along_axis(lambda x:x*w_vec,0,x)) + ridge)
            beta_var_mat[i] = np.diag(sigma)
        return beta_mat[start:end],beta_var_mat[start:end],hat_diagonals[start:end]

