import scipy.special as ss
import scipy.optimize as so
from scipy.stats import norm as dnorm
import numpy as np


class beta_optim_calculator():
    def __init__(self,raw_count,designmatrix,rowsForOptim,rowStable,
                    normalizationFactors,alpha_hat,
                    betaMatrix,betaSE,
                    beta_mat,
                    mu,logLike,minmu=0.5):
        self.K = np.array(raw_count)
        self.designmatrix = designmatrix
        self.s = normalizationFactors
        self.beta_mat = beta_mat
        self.rowsForOptim = rowsForOptim
        self.rowStable = rowStable
        self.alpha = alpha_hat
        self.betaMatrix = betaMatrix
        self.betaSE = betaSE
        self.mu = mu
        self.logLike = logLike
        self.minmu = minmu

    def logNegativeBinomial(self , K , alpha , mu):
        return ss.loggamma(K +1/alpha)- ss.loggamma(1/alpha)  - K * np.log(mu + 1/alpha)- (1/alpha) * np.log(1+alpha*mu) 
    
    def objectiveFn(self,p,Lambda,x,nf,k,alpha):
        mu_row = nf * (2**(x@p))
        logLikeVector = self.logNegativeBinomial(k,alpha,mu_row)
        logLike = logLikeVector.sum()
        logPrior = dnorm.logpdf(p,0,np.sqrt(1/Lambda))
        negLogPost = -logLike-logPrior
        if not np.isfinite(negLogPost):
            return 1e300
        else:
            return negLogPost

    def fitBetaOptim(self,start,end):
        x = self.designmatrix
        Lambda= 1e-6*np.ones(x.shape[1])
        lambdaNatLogScale = Lambda / (np.log(2)**2)
        large = 30
        op_args = self.rowsForOptim[start:end]
        for row in op_args:
            if self.rowStable[row] and (abs(self.betaMatrix[row])<large).all():
                betaRow = self.betaMatrix[row]
            else :
                betaRow = self.beta_mat[row]
            nf = self.s
            k = self.K[row]
            alpha = self.alpha[row]
            o = so.minimize(self.objectiveFn,betaRow,args=(Lambda,x,nf,k,alpha),method='L-BFGS-B',bounds=((-large,large),(-large,large)))
            ridge = np.diag(lambdaNatLogScale)

            self.betaMatrix[row] = o.x
            mu_row = nf * 2**(x@o.x)
            self.mu[row] = mu_row
            mu_row = max(self.minmu,mu_row)
            w = np.diag(1/(1/mu_row+alpha))
            xtwx = x.T@w@x
            xtwxRidgeInv = np.linalg.pinv(xtwx+ridge)
            sigma = xtwxRidgeInv@xtwx@xtwxRidgeInv
            self.betaSE[row] = np.log2(np.exp(1)) * np.sqrt(max(np.diag(sigma),0))
            logLikeVector = self.logNegativeBinomial(k,alpha,mu_row)
            self.logLike[row] = logLikeVector.sum()

        return (np.take(self.betaMatrix,op_args,axis=0) , np.take(self.betaSE,op_args,axis=0),
        np.take(self.mu,op_args,axis=0),np.take(self.logLike,op_args,axis=0))

    
