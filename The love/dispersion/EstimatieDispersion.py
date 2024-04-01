
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as smxswl
import scipy.special as ss
import scipy.optimize as so
import Estimate_Size_Factor as ESF
import matplotlib.pyplot as plt
import math
from scipy.misc import derivative
'''
countable: m*n matrix, m: number of genes, n: number of samples 
design matrix: n*k matrix, k: number of coefficients
size factors: n*1 matrix
'''

# rough method-of-moments estimate of dispersion

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
    #print(singleResponse_res.params)

    # Initialize mean
    para = singleResponse_res.params
    Mu = size_factors.T*(np.exp(designmatrix@para.T))
    mu = Mu.tolist()[0]
    return mu

class alpha_calculator():
    def __init__(self,raw_count,mu,ori_alpha,designmatrix,line):
        self.K = raw_count
        self.mu = mu
        self.oa = ori_alpha
        self.dig = ss.digamma
        self.em = 0.57721
        self.designmatrix = designmatrix
        self.l = line
        self.op_list = []
    
    
    
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
        ##print('mg',math.gamma(K +1/alpha))
        #print('input for mass "{},{},{}"'.format(K,mu,alpha))
        return math.lgamma(K +1/alpha)- math.lgamma(K+1)- math.lgamma(1/alpha)  - (1/alpha) * math.log(1+alpha*mu) + K * math.log(alpha*mu / (1+alpha*mu))
    def dlogNegativeBinomial(self , K , mu , alpha):
        #alpha_neg2 * sum(Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1.0 + mu*alpha, -1) - digamma(y + alpha_neg1) + y * pow(mu + alpha_neg1, -1));
        return (1/alpha**2)*(ss.digamma(1/alpha) + np.log(mu * alpha + 1) - mu * alpha * (1/(1+mu*alpha)) - ss.digamma(K + 1/alpha) + K * (1/(mu + (1/alpha))))
    def ddlogNegativeBinomial(self,K,mu,alpha):
        #-2 * R_pow_di(alpha, -3) * sum(Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1 + mu*alpha, -1) - digamma(y + alpha_neg1) + y * pow(mu + alpha_neg1, -1)) + alpha_neg2 * sum(-1 * alpha_neg2 * Rf_trigamma(alpha_neg1) + pow(mu, 2) * alpha * pow(1 + mu*alpha, -2) + alpha_neg2 * trigamma(y + alpha_neg1) + alpha_neg2 * y * pow(mu + alpha_neg1, -2));
        -2 * (1/alpha**3) * (ss.digamma(1/alpha) + np.log(1+mu*alpha) - mu*alpha*(1/(1+mu*alpha)) - ss.digamma(K+(1/alpha)) + K*(1/mu+(1/alpha))) + (1/alpha**2) * (-(1/alpha**2) * ss.polygamma(1,1/alpha) + mu**2 * alpha * (1/(1+mu*alpha)**2) + (1/alpha**2) * ss.polygamma(1,K + (1/alpha)) + (1/alpha**2) * K * (1/(mu+(1/alpha))**2)) 
    def logLike(self , alpha):
        res = 0
        for j in range(len(self.mu)):
            #print('rrrrrrr {}'.format(j))
            #print('input for totl "{},{},{}"'.format(self.K[j],self.mu[j],alpha))
            res += self.logNegativeBinomial(self.K[j] , self.mu[j] , alpha)
        return res
    def dloglike(self,alpha):
        res = 0
        for j in range(len(self.mu)):
            #print('rrrrrrr {}'.format(j))
            #print('input for totl "{},{},{}"'.format(self.K[j],self.mu[j],alpha))
            res += self.dlogNegativeBinomial(self.K[j] , self.mu[j] , alpha)
        return res
    def ddloglike(self,alpha):
        res = 0
        for j in range(len(self.mu)):
            #print('rrrrrrr {}'.format(j))
            #print('input for totl "{},{},{}"'.format(self.K[j],self.mu[j],alpha))
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
    def final_Loglike_CR(self,logalpha):
        alpha = np.exp(logalpha)
        return self.logLike(alpha) + self.matrix_part(alpha)
    def final_dev(self,logalpha):
        alpha = np.exp(logalpha)
        return (self.dloglike(alpha) + self.dmatrix(alpha)) * alpha
    def final_do_dev(self,logalpha):
        alpha = np.exp(logalpha)
        #res = ((ll_part + cr_term) * R_pow_di(alpha, 2) + dlog_posterior(log_alpha, y, mu, x_orig, log_alpha_prior_mean, log_alpha_prior_sigmasq, false, weights, useWeights, weightThreshold, useCR)) + prior_part;
        return (self.ddloglike(alpha) + self.ddmatrix(alpha)) * alpha**2 + self.dloglike(alpha) 
    
    
    
    
    ## Helpful Functions For Optimization
    def convex_checker(self,f):
        # a = 1e-8
        # b = 12
        # if f((a+b)/2) >= (f(a)+f(b))/2:
        #     return True
        # return False
        fp = lambda x:derivative(f, x,dx=x * 1e-2)
        fpp = lambda x:derivative(fp,x,dx=x * 1e-2)
        fppx = np.zeros(11)
        for i in range(11):
            fppx[i] = fpp(i+1e-2)
        if (fppx>=0).all():
            return True
        # return False
        # return True
    
    ## Gradient Descent
    def gradient_descent(self,f_prime,start=1,tol = 1e-5,h=1e-2,top_ir=20000, error_tol = 1e-2): #20000
        step_num = 0
        critical = self.oa
        pre_critical = 0
        while step_num <= top_ir:
            # if step_num % 500 == 0:
            #     print('round {}'.format(step_num) )
            if critical <= 0:
                critical = -critical+1e-1
            pre_critical = critical
            #print('ppp {}'.format(pre_critical))
            critical = pre_critical + h * f_prime(pre_critical)
            step_num += 1
            if abs(pre_critical - critical) <= tol and critical >= 0:
                break
        return critical
    def calculate(self):
        return self.gradient_descent(self.final_dev)
    
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
                kappa *= 1.1
                if it_accpet % 5 == 0:
                    kappa = kappa / 2
            else:
                kappa = kappa / 2 
        return np.exp(a)

class MLE_estimator(alpha_calculator):
    def __init__(self , df , compare,raw_count,mu,ori_alpha,designmatrix,line):
        
        self.raw_count = df
        self.compare = compare
        self.normalized_count = ESF.estimateSizeFactor(self.raw_count)[0].to_numpy()
        self.size_factors = ESF.estimateSizeFactor(self.raw_count)[1].to_numpy()
        
    def designmatrix(self):
        
        pass
    def estimate(self,name):
        
        pass
    def estimate_all(self):
        pass   
    def __format__(self, __format_spec: str) -> str:
        pass
    
def estimateDispersionParameterGenewise(raw_count, designmatrix , normalized_count ,size_factors,start,end):
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
            if roughDispEst[idx] >= 1e-8:
                singleGeneResponse = normalized_count_np[idx]
                mu = initialMu(singleGeneResponse, designmatrix, roughDispEst, idx, size_factors)
                k = raw_count.to_numpy()[idx]
                calculator = alpha_calculator(k,mu,np.log(roughDispEst)[idx],designmatrix,0.1)
                single = calculator.line_gradient()
                est_genewise_alpha_dict[normalized_count.index[idx]] = min(max(single,1e-8) , 10)
            else:
                est_genewise_alpha_dict[normalized_count.index[idx]] = 1e-8
        else:
            est_genewise_alpha_dict[normalized_count.index[idx]] = "NA"
    return pd.DataFrame(est_genewise_alpha_dict.values(),est_genewise_alpha_dict.keys())
    # queue.put(est_genewise_alpha_dict)
# count_table = pd.read_csv('maqsAllCounts.csv', index_col=0)
# Norm_NM_000014_4 = pd.DataFrame({"JOA1": [0],"JOA2": [4.697854],"JOA3": [3.063885],"JOA4": [0],"JOA5": [0.937649],"JOB1": [0],"JOB2": [0],"JOB3": [0.620733],"JOB4": [0.937502],"JOB5": [1.656273]}, index=["NM_000014_4"])
# Norm_NM_000014_4_T = np.transpose(Norm_NM_000014_4)
# Norm_NM_000014_4_T.insert(0, "type", ['univ', 'univ', 'univ', 'univ', 'univ', 'brain', 'brain', 'brain', 'brain', 'brain'])
# response, designmatrix = dmatrices( "NM_000014_4~ C(type)", Norm_NM_000014_4_T, return_type="matrix")  
##print(estimateDispersionParameterGenewise(count_table, designmatrix)
# )