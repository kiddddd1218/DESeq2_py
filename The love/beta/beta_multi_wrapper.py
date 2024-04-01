from Beta import *
from betaOptim import *
from multi_process import multi
import Estimate_Size_Factor as ESF
import pandas as pd
from patsy import dmatrices
from scipy.stats import norm


class beta_wrapper:
    def __init__(self):
        self.raw_count = None
        self.raw_countNZ = None
        self.designmatrix = None
        self.size_factors = None
        self.normalized_count = None
        self.dispersion = None
        self.beta = None
        self.beta_var = None
        self.nbnom = None
        self.beta_int_cal = None
        self.beta_optim_cal = None
        self.inter_res = None
        self.beta_optim_res = None
        self.rowForOptim = None
        self.betaMatrix = None
        self.betaSE = None
        self.mu = None
        self.logLike = None
        self.justInter = True

    def data_prep_beta_int(self):
        self.raw_count = pd.read_csv('maqsAllCounts.csv', index_col=0)
        Norm_NM_000014_4 = pd.DataFrame({"JOA1": [0], "JOA2": [4.697854], "JOA3": [3.063885], "JOA4": [0], "JOA5": [0.937649], "JOB1": [
                                        0], "JOB2": [0], "JOB3": [0.620733], "JOB4": [0.937502], "JOB5": [1.656273]}, index=["NM_000014_4"])
        Norm_NM_000014_4_T = np.transpose(Norm_NM_000014_4)
        Norm_NM_000014_4_T.insert(0, "type", [
            'univ', 'univ', 'univ', 'univ', 'univ', 'brain', 'brain', 'brain', 'brain', 'brain'])
        feckless, self.designmatrix = dmatrices(
            "NM_000014_4~ C(type)", Norm_NM_000014_4_T, return_type="matrix")
        self.designmatrix = np.array(self.designmatrix)
        self.dispersion = pd.read_csv('dispersion/data/Ultradisp.csv', index_col='RefSeq')[
            self.raw_count.sum(axis=1) != 0].to_numpy()

        normalized_count, self.size_factors = ESF.estimateSizeFactor(
            self.raw_count)
        self.normalized_count = normalized_count[self.raw_count.sum(axis=1) != 0]
        self.raw_countNZ = self.raw_count[self.raw_count.sum(axis=1) != 0]
        

        y = np.log(normalized_count+.1)
        if np.linalg.matrix_rank(self.designmatrix) == self.designmatrix.shape[1]:
            q, r = np.linalg.qr(self.designmatrix)
            self.beta = np.linalg.solve(r, (q.T)@(y.T)).T
        else:
            self.beta = np.ones((y.shape[0], self.designmatrix.shape[1]))

        self.beta_var = np.apply_along_axis(self.var, 1, self.beta)

        self.beta_int_cal = beta_calculator(self.raw_countNZ, self.designmatrix, disp=self.dispersion,
                                            variance_beta=self.beta_var, size_factors=self.size_factors,
                                            beta_init=self.beta)

    def var(self, b):
        real_b = b[b < 10]
        if len(real_b) == 0:
            return 1e6
        else:
            return np.quantile(real_b, 0.95, axis=0) / norm.ppf(1-0.05/2)

    def data_prep_beta_optim(self):
        self.beta_mat = np.concatenate(
            [self.inter_res[i][0] for i in range(len(self.inter_res))])
        self.beta_var = np.concatenate(
            [self.inter_res[i][1] for i in range(len(self.inter_res))])
        self.hat_diagonals_Res = np.concatenate(
            [self.inter_res[i][2] for i in range(len(self.inter_res))])
        mu = self.size_factors.to_numpy().squeeze(
        )*(np.exp(self.designmatrix@(self.beta_mat.T)).T)

        self.logLike = (self.beta_int_cal.logNegativeBinomial(
            self.raw_countNZ.to_numpy(), self.dispersion, mu)).sum(axis=1)
        rowStable = np.apply_along_axis(lambda row: not (
            np.isnan(row)).any(), axis=1, arr=self.beta_mat)
        rowVarPositive = np.apply_along_axis(
            lambda row: not (row <= 0).any(), axis=1, arr=self.beta_var)
        self.betaMatrix = np.log2(np.exp(1))*self.beta_mat
        self.betaSE = np.log2(
            np.exp(1))*np.sqrt(np.where(self.beta_var >= 0, self.beta_var, 0))
        self.rowForOptim = np.union1d(
            np.where((rowStable == False)), np.where((rowVarPositive == False)))
        self.beta_optim_cal = beta_optim_calculator(self.raw_countNZ, self.designmatrix,
                                                    self.rowForOptim, rowStable,
                                                    self.size_factors, self.dispersion,
                                                    self.betaMatrix, self.betaSE,
                                                    self.beta_mat,
                                                    mu, self.logLike, minmu=0.5)

    def beta_int_func(self, start, end):
        return self.beta_int_cal.fitBeta(start, end)

    def beta_optim_func(self, start, end):
        return self.beta_optim_cal.fitBetaOptim(start, end)

    # def main1(self):
    #     self.data_prep_beta_int()
    #     beta_int_multi = multi(self.raw_countNZ.shape[0]-1,self.beta_int_func)
    #     self.inter_res = beta_int_multi.run()
    #     self.data_prep_beta_optim()
        # if len(self.rowForOptim) > 0:
        #     beta_optim_multi = multi(self.rowForOptim.shape[0]-1,self.beta_optim_func)
        #     self.beta_optim_res = beta_optim_multi.run()
        #     np.put(self.betaMatrix,self.rowForOptim,np.concatenate([self.beta_optim_res[i][0] for i in range(len(self.beta_optim_res))]))
        #     np.put(self.betaSE,self.rowForOptim,np.concatenate([self.beta_optim_res[i][1] for i in range(len(self.beta_optim_res))]))
        #     np.put(self.mu,self.rowForOptim,np.concatenate([self.beta_optim_res[i][2] for i in range(len(self.beta_optim_res))]))
        #     np.put(self.logLike,self.rowForOptim,np.concatenate([self.beta_optim_res[i][3] for i in range(len(self.beta_optim_res))]))
        #     pd.DataFrame(data =self.betaMatrix,index=self.raw_count.index).to_csv('dispersion/betaMatrix.csv')
        #     pd.DataFrame(data =self.betaSE,index=self.raw_count.index).to_csv('dispersion/betaSE.csv')
        #     # pd.DataFrame(data =self.mu,index=self.raw_count.index).to_csv('dispersion/mu.csv')
        #     # pd.DataFrame(data =self.logLike,index=self.raw_count.index).to_csv('dispersion/logLike.csv')

    def main(self):
        self.data_prep_beta_int()
        beta_int_multi = multi(self.raw_countNZ.shape[0]-1, self.beta_int_func)
        self.inter_res = beta_int_multi.run()
        self.data_prep_beta_optim()
        if len(self.rowForOptim) > 0:
            beta_optim_multi = multi(
                self.rowForOptim.shape[0]-1, self.beta_optim_func)
            self.beta_optim_res = beta_optim_multi.run()
            np.put(self.betaMatrix, self.rowForOptim, np.concatenate(
                [self.beta_optim_res[i][0] for i in range(len(self.beta_optim_res))]))
            np.put(self.betaSE, self.rowForOptim, np.concatenate(
                [self.beta_optim_res[i][1] for i in range(len(self.beta_optim_res))]))
            np.put(self.mu, self.rowForOptim, np.concatenate(
                [self.beta_optim_res[i][2] for i in range(len(self.beta_optim_res))]))
            np.put(self.logLike, self.rowForOptim, np.concatenate(
                [self.beta_optim_res[i][3] for i in range(len(self.beta_optim_res))]))
            # pd.DataFrame(data =self.mu,index=self.raw_count.index).to_csv('dispersion/mu.csv')
            # pd.DataFrame(data =self.logLike,index=self.raw_count.index).to_csv('dispersion/logLike.csv')
        else:
            print('No further optimization needed')

        betaM = pd.DataFrame(data=self.betaMatrix,
                             index=self.raw_countNZ.index)
        pd.DataFrame(data=betaM, index=self.raw_count.index).to_csv(
            'beta/data/betaMatrix.csv')
        betaSE = pd.DataFrame(data=self.betaSE, index=self.raw_countNZ.index)
        pd.DataFrame(data=betaSE, index=self.raw_count.index).to_csv(
            'beta/data/betaSE.csv')
        betalogLike = pd.DataFrame(
            data=self.logLike, index=self.raw_countNZ.index)
        pd.DataFrame(data=betalogLike, index=self.raw_count.index).to_csv(
            'beta/data/betalogLike.csv')

    def justIntercept(self):
        alpha = self.dispersion
        betaConv = np.ones(self.raw_countNZ.shape[0])
        betaIter = np.ones(self.raw_countNZ.shape[0])
        betaMatrix = np.log2(self.normalized_count.mean(axis=1))
        mu = (np.array(2**(betaMatrix)).reshape(-1,1))@(self.size_factors.T)
        print(self.normalized_count)# self.size_factors['JOA1']*betaMatrix
        return
        logLikeMat = self.beta_int_cal.logNegativeBinomial(self.raw_countNZ, alpha, mu)
        logLike = logLikeMat.sum(axis=1)
        modelMatrix = self.designmatrix
        w = np.linalg.pinv(np.linalg.pinv(mu) + alpha.T)
        
        xtwx = np.array(w.sum(axis=1)).reshape(-1,1)
        sigma = 1/xtwx
        betaSE = np.log2(np.exp(1)) * np.sqrt(sigma)
        hat_diagonals = w / xtwx 
        return mu , betaMatrix

if __name__ == '__main__':
    wrapper = beta_wrapper()
    wrapper.main()
    print(wrapper.hat_diagonals_Res)
    # print(res[0][0])