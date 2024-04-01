

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import scipy.special as ss
import scipy.optimize as so
import Estimate_Size_Factor as ESF
import matplotlib.pyplot as plt
import math
from scipy.misc import derivative
from statsmodels import robust
import matplotlib.pyplot as plt
import time 

def linearModelMu(y, x):
    '''
    y: normalized counts (m*n)
    x: model matrix (standard design matrix)(n*k)

    return mean of the counts
    '''
    q, r = np.linalg.qr(x)
    rinv = np.linalg.inv(r)
    return (y@q)@((x@rinv).transpose()) 

def pmax(mat, num):
    mat_rtn = mat.copy()
    if len(mat.shape) == 1:
        for i in range(mat.shape[0]):
            if mat[i] < num:
                mat_rtn[i] = num
    else:
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                if mat[row][col] < num:
                    mat_rtn[row][col] = num
    return mat_rtn

def roughDispEstimate(normalized_count, designmatrix , size_factor):
    '''
    return initial value of alpha
    '''
    mu = linearModelMu(normalized_count, designmatrix)
    mu = pmax(mu, 1)
    xim = (1/size_factor).mean()
    bv = normalized_count.var(axis=1)
    bm = normalized_count.mean(axis=1)
    nrow, ncol = designmatrix.shape
    est1 = np.sum(((normalized_count-mu)**2-mu)/mu**2, axis=1)/(nrow-ncol)
    est2 = (bv-xim*bm)/bm**2
    est = np.where(est1 > est2 , est2, est1)
    return est


def initialMu(singleResponse, designmatrix, roughDispEst, idx, size_factors):
    '''
    singleResponse: (n,1) 
    designmatrix: (n,k)
    roughDispEst: (m,) initial dispersion parameters for all genes 
    idx: integer denotes the index of the gene
    size_factors: (n,1)

    return initialized mean
    '''
    # Using GLM to estimate the coefficient
    singleResponse_res = sm.GLM(singleResponse, designmatrix, family=sm.families.NegativeBinomial(alpha=roughDispEst[idx])).fit()

    # Initialize mean
    para = singleResponse_res.params
    Mu = size_factors.T*(np.exp(designmatrix@para.T))
    mu = Mu.tolist()[0]
    return mu


class alpha_calculator():
    def __init__(self,raw_count,mu,logalpha_ori,designmatrix,logalpha_prior,variance_d):
        self.K = raw_count
        self.mu = mu
        self.oa = logalpha_ori
        self.dig = ss.digamma
        self.em = 0.57721
        self.designmatrix = designmatrix
        self.op_list = []
        self.logalpha_prior = logalpha_prior
        self.var_d = variance_d
    
    ## Loglikelihood Functions
    def W(self,alpha):
        l = [1/(1/self.mu[j]+alpha) for j in range(len(self.mu))]
        return np.diag(l)
    def dw(self,alpha):
        l = [-1/(1/self.mu[j]+alpha)**2 for j in range(len(self.mu))]
        return np.diag(l)
    def ddw(self,alpha):
        l = [2/(1/self.mu[j]+alpha)**3 for j in range(len(self.mu))]
    def logNegativeBinomial(self , K , mu , alpha):
        #lgamma(y + alpha_neg1) - Rf_lgammafn(alpha_neg1) - y * log(mu + alpha_neg1) - alpha_neg1 * log(1.0 + mu * alpha));
        return math.lgamma(K +1/alpha)- math.lgamma(1/alpha)  - K * math.log(mu + 1/alpha)- (1/alpha) * math.log(1+alpha*mu) 
    def dlogNegativeBinomial(self , K , mu , alpha):
        #alpha_neg2 * sum(Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1.0 + mu*alpha, -1) - digamma(y + alpha_neg1) + y * pow(mu + alpha_neg1, -1));
        return (1/alpha**2)*(ss.digamma(1/alpha) + np.log(mu * alpha + 1) - mu * alpha * (1/(1+mu*alpha)) - ss.digamma(K + 1/alpha) + K * (1/(mu + (1/alpha))))
    def ddlogNegativeBinomial(self,K,mu,alpha):
        #-2 * R_pow_di(alpha, -3) * sum(Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1 + mu*alpha, -1) - digamma(y + alpha_neg1) + y * pow(mu + alpha_neg1, -1)) + alpha_neg2 * sum(-1 * alpha_neg2 * Rf_trigamma(alpha_neg1) + pow(mu, 2) * alpha * pow(1 + mu*alpha, -2) + alpha_neg2 * trigamma(y + alpha_neg1) + alpha_neg2 * y * pow(mu + alpha_neg1, -2));
        return -2 * (1/alpha**3) * (ss.digamma(1/alpha) + np.log(1+mu*alpha) - mu*alpha*(1/(1+mu*alpha)) - ss.digamma(K+(1/alpha)) + K*(1/mu+(1/alpha))) + (1/alpha**2) * (-(1/alpha**2) * ss.polygamma(1,1/alpha) + mu**2 * alpha * (1/(1+mu*alpha)**2) + (1/alpha**2) * ss.polygamma(1,K + (1/alpha)) + (1/alpha**2) * K * (1/(mu+(1/alpha))**2)) 
    def logLike(self , alpha):
        res = 0
        for j in range(len(self.mu)):
            res += self.logNegativeBinomial(self.K[j] , self.mu[j] , alpha)
        return res
    def dloglike(self,alpha):
        res = 0
        for j in range(len(self.mu)):
            res += self.dlogNegativeBinomial(self.K[j] , self.mu[j] , alpha)
        return res
    def ddloglike(self,alpha):
        res = 0
        for j in range(len(self.mu)):
            res += self.ddlogNegativeBinomial(self.K[j] , self.mu[j] , alpha)
        return res
    def matrix_part(self,alpha):
        return -(1/2) * math.log(np.linalg.det(self.designmatrix.T @ self.W(alpha) @ self.designmatrix)) 
    def dmatrix(self,alpha):
        b = self.designmatrix.T @ self.W(alpha) @ self.designmatrix
        db = self.designmatrix.T @ self.dw(alpha) @ self.designmatrix
        ddetb = (np.linalg.det(b) * np.trace(np.linalg.inv(b) @ db))
        return -0.5 * ddetb / np.linalg.det(b) 
    def ddmatrix(self,alpha):
        b = self.designmatrix.T @ self.W(alpha) @ self.designmatrix
        b_i = np.linalg.inv(b)
        db = self.designmatrix.T @ self.dw(alpha) @ self.designmatrix
        ddb = self.designmatrix.T @ self.ddw(alpha) @ self.designmatrix
        ddetb = (np.linalg.det(b) * np.trace(np.linalg.inv(b) @ db))
        #d2detb = ( det(b) * (R_pow_di(trace(b_i * db), 2) - trace(b_i * db * b_i * db) + trace(b_i * d2b)) );
        d2detb = np.linalg.det(b) * ((np.trace(b_i @ db))**2 - np.trace(b_i @ db @ b_i @ db) + np.trace(b_i @ ddb))
        #cr_term = 0.5 * R_pow_di(ddetb/det(b), 2) - 0.5 * d2detb / det(b);
        return 0.5 * (ddetb/np.linalg.det(b))**2 - 0.5 * d2detb/np.linalg.det(b)
    def LAMBDA(self,logalpha):
        return -0.5*((logalpha-self.logalpha_prior)**2) / self.var_d
    def dLAMBDA(self,logalpha):
        return -(logalpha-self.logalpha_prior) / self.var_d
    def ddLAMBDA(self):
        #prior_part = -1.0/log_alpha_prior_sigmasq;
        return -1/self.var_d  
    def final_Loglike_CR(self,logalpha):
        alpha = np.exp(logalpha)
        return self.logLike(alpha) + self.matrix_part(alpha) + self.LAMBDA(logalpha)
    def final_dev(self,logalpha):
        alpha = np.exp(logalpha)
        return (self.dloglike(alpha) + self.dmatrix(alpha)) * alpha + self.dLAMBDA(logalpha)
    def final_do_dev(self,logalpha):
        alpha = np.exp(logalpha)
        #res = ((ll_part + cr_term) * R_pow_di(alpha, 2) + dlog_posterior(log_alpha, y, mu, x_orig, log_alpha_prior_mean, log_alpha_prior_sigmasq, false, weights, useWeights, weightThreshold, useCR)) + prior_part;
        return (self.ddloglike(alpha) + self.ddmatrix(alpha)) * alpha**2 + self.dloglike(alpha) + self.ddLAMBDA()
    
    def line_gradient(self,top_ir=100, error_tol = 1e-8): #20000
        epsilon = 1e-4
        kappa_0 = 1
        kappa = kappa_0
        a = self.oa
        lp = self.final_Loglike_CR(a)
        dlp = self.final_dev(a)
        change = -1
        it = 0
        it_accpet = 0
        while it <= top_ir:
            it += 1
            a_propose = a + kappa * dlp
            if a_propose < -30:
                kappa = (-30 - a) / dlp
            if a_propose > 10:
                kappa = (10-a)/dlp
            theta_kappa = -self.final_Loglike_CR(a + kappa * dlp)
            theta_hat_kappa = -lp - kappa * epsilon * dlp ** 2
            if theta_kappa <= theta_hat_kappa:
                it_accpet += 1
                a = a + kappa * dlp
                lpnew = self.final_Loglike_CR(a)
                change = lpnew - lp
                if change < error_tol:
                    lp = lpnew
                    break
                if a < np.log(1e-9):
                    break
                lp = lpnew
                dlp = self.final_dev(a)
                kappa = min(1.1*kappa,kappa_0)
                if it_accpet % 5 == 0:
                    kappa = kappa / 2
            else:
                kappa = kappa / 2 
        return np.exp(a)
    
    def draw(self):     
        x = np.linspace(0.1,5,1000)
        y = [self.final_Loglike_CR(i) for i in x]
        plt.plot(x,y)
        plt.show()
        
class trend():
    def __init__(self,gw_dis,basemean) -> None:
        self.gwdisNZ = gw_dis
        self.meanNZ = basemean
        self.meanNZ_with1 = self.meanNZ_with_1()
        self.meanNZ_with1_forfit = self.meanNZ_with1[self.gwdisNZ>=1e-6]
        self.gwdisNZ_forfit = self.gwdisNZ[self.gwdisNZ>=1e-6]
        self.beta = None
        self.method = 'para'
        
        
    def meanNZ_with_1(self):
        '''
        Add one column to meanNZ
        '''
        meanNZ_with_1_np = np.array([np.ones(len(self.meanNZ)),self.meanNZ]).T
        meanNZ_with_1_df = pd.DataFrame(meanNZ_with_1_np,index=self.meanNZ.index)
        
        return meanNZ_with_1_df
    
    def trend_para(self,max_iter=10):
        iter = 0
        beta = np.array([.1,1])
        while True: 
            fitted = beta[1] / self.meanNZ_with1_forfit[1] + beta[0]
            fitted = self.gwdisNZ_forfit / fitted 
            self.gwdisNZ_forfit = self.gwdisNZ_forfit[fitted<15] 
            self.gwdisNZ_forfit = self.gwdisNZ_forfit[fitted>1e-4]
            self.meanNZ_with1_forfit = self.meanNZ_with1_forfit[fitted<15] 
            self.meanNZ_with1_forfit = self.meanNZ_with1_forfit[fitted>1e-4]
            gamma_model = sm.GLM(self.gwdisNZ_forfit, 1/self.meanNZ_with1_forfit, family=sm.families.Gamma(link=sm.families.links.identity()))#
            gamma_res = gamma_model.fit(start_params=beta).params
            old_beta = beta
            beta = gamma_res
            if (beta <= 0).any():
                print('Para fit failed, change to local fit.')
                self.method = 'loc'
                break 
            
            if ((np.log(beta/old_beta))**2).sum() < 1e-6:
                break
                                             
            iter += 1
            if iter > max_iter: 
                print('Para fit failed, change to local fit.')
                self.method = 'loc'
                break
            
        self.beta = beta
        
    def trend_loc(self):
        colA = self.mu_bar_res
        colA[0] = np.log(colA[0])
        colB = np.log(self.gwdis_res)
        
        smoothed = sm.nonparametric.lowess(exog=colA[0], endog=colB, frac=0.2,return_sorted=False)
        smoothed = pd.Series(smoothed,index=colA.index)
       
        return smoothed
        ## kernel
        # diff = point - colA
        
        # kernel = diff.apply(lambda x: np.exp((x@x.T)/(-2*tau**2)),axis=1)
        # print(self.bm)
        # wt = np.diag(self.bm)
        # #print('shapewt {}, shape_mu {},shape gw {}'.format(wt.shape , colA.shape , colB.shape))
        # W = np.linalg.pinv(wt@colA)@wt@colB
        # W = np.linalg.pinv((colA.T)@colA)@(colA.T)@(colB)
        # W = np.array([1.4682599,-9.6246442])
        #return colA @ W
    
    def trend_func(self):
        if self.method == 'para':
            trend = self.beta[1] / self.meanNZ_with1[1] + self.beta[0]
            # dispInit = np.where(self.gwdisNZ > 0.1 * trend , self.gwdisNZ, trend)
            # dispInit_df = pd.Series(dispInit,index=trend.index)
            return np.log(trend)
        if self.method == 'loc':
            return self.trend_loc()
        
def estimateDispersionParameterGenewise_with_prior(raw_count, designmatrix , normalized_count ,size_factors,logalpha_prior,var_d,start,end):
    '''
    *** expedient input since gradient descent is at a low efficiency
    input:
    normalized_count, size_factors are dataframes from function estimateSizeFactor()

    output: 
    est_genewise_alpha_dict: dictionary with gene name as keys 
        and genewise dispersion para estimations as values
    '''
    size_factors = size_factors.to_numpy()
    normalized_count_np = normalized_count.to_numpy()

    roughDispEst =  np.array(roughDispEstimate(normalized_count_np, designmatrix,size_factors))
    roughDispEst = np.where(roughDispEst > 1e-8 , roughDispEst , 1e-8)
    roughDispEst = np.where(roughDispEst < 10 , roughDispEst , 10)
    est_genewise_alpha_dict = {}
    for idx in range(start,end):
        if idx % 1000 == 0:
            print('gene',idx)
        if (normalized_count_np[idx] != 0).any():
            G_name = raw_count.index[idx]
            lalpha_prior = logalpha_prior[G_name]
            singleGeneResponse = normalized_count_np[idx]
            mu = initialMu(singleGeneResponse, designmatrix, roughDispEst, idx, size_factors)
            k = raw_count.to_numpy()[idx]
            calculator = alpha_calculator(k,mu,np.log(roughDispEst)[idx],designmatrix,lalpha_prior,var_d)
            single = calculator.line_gradient()
            dispersion_proposal =  min(max(single,1e-8),10)
            est_genewise_alpha_dict[normalized_count.index[idx]] = min(max(single,1e-8),10)
        else:
            est_genewise_alpha_dict[normalized_count.index[idx]] = "NA"
    return pd.DataFrame(est_genewise_alpha_dict.values(),est_genewise_alpha_dict.keys())