
import Estimate_Size_Factor as ESF
from EstimateDispersionWithPrior import *
from EstimatieDispersion import *
from scipy.stats import norm
from multi_process import multi

class Dispersion:
    def __init__(self) -> None:
        self.raw_count = None
        self.designmatrix = None
        self.normalized_count = None
        self.size_factors = None
        self.logalpha = None
        self.variance_d = None
        self.s_lr = None
        self.length = None
        self.basemean = None
        self.gwdis = None
        self.map = None
        self.final_res = None
        self.gwdispNZ = None
    def pre_data_MLE(self):
        raw_count = pd.read_csv('maqsAllCounts.csv', index_col=0)
        Norm_NM_000014_4 = pd.DataFrame({"JOA1": [0],"JOA2": [4.697854],"JOA3": [3.063885],"JOA4": [0],"JOA5": [0.937649],"JOB1": [0],"JOB2": [0],"JOB3": [0.620733],"JOB4": [0.937502],"JOB5": [1.656273]}, index=["NM_000014_4"])
        Norm_NM_000014_4_T = np.transpose(Norm_NM_000014_4)
        Norm_NM_000014_4_T.insert(0, "type", ['univ', 'univ', 'univ', 'univ', 'univ', 'brain', 'brain', 'brain', 'brain', 'brain'])
        feckless, designmatrix = dmatrices( "NM_000014_4~ C(type)", Norm_NM_000014_4_T, return_type="matrix")  
        designmatrix = np.array(designmatrix)
        

        normalized_count, size_factors =  ESF.estimateSizeFactor(raw_count)
        self.raw_count , self.designmatrix, self.normalized_count, self.size_factors = raw_count , designmatrix, normalized_count,size_factors
        self.basemean = self.normalized_count.mean(axis=1)
        self.length = self.raw_count.shape[0]-1

    def pre_data_MAP(self):
        raw_countC = self.raw_count
        raw_countC['sum'] = raw_countC.sum(axis=1)
        self.gwdispNZ = pd.to_numeric(self.gwdis[raw_countC['sum'] != 0])
        basemeanNZ = self.basemean[raw_countC['sum'] != 0]
        tr = trend(self.gwdispNZ,basemeanNZ)

        tr.trend_para()
        loggw = np.log(self.gwdispNZ)
        self.logalpha = tr.trend_func()
        res = loggw-self.logalpha
        self.s_lr = robust.scale.mad(res[res>np.log(1e-6)])
        self.variance_d = max(self.s_lr**2-ss.polygamma(1,4) , 0.25)  
        
    def func_MAP(self,x,y):
        return estimateDispersionParameterGenewise_with_prior(self.raw_count,self.designmatrix,self.normalized_count,self.size_factors,self.logalpha,self.variance_d,x,y)

    def func_MLE(self,x,y):
        return estimateDispersionParameterGenewise(self.raw_count,self.designmatrix,self.normalized_count,self.size_factors,x,y)

    def MLE(self):
        self.pre_data_MLE()
        mle_multi = multi(self.length,self.func_MLE)
        self.gwdis = pd.concat(mle_multi.run())[0]

    def MAP(self):
        self.pre_data_MAP()
        map_multi = multi(self.length,self.func_MAP)
        self.map = pd.concat(map_multi.run())

    def main(self):
        self.MLE()
        self.MAP()
        difference = np.log(self.gwdispNZ) - self.logalpha
        final_res = self.map.loc[difference.index].squeeze()
        final_res = np.where(difference.squeeze() >= 2*self.s_lr,self.gwdispNZ.squeeze(),final_res)
        final_res = pd.DataFrame(index=difference.index,data=final_res)
        ultra_res = pd.DataFrame(index=self.raw_count.index,data=final_res)
        ultra_res.to_csv('dispersion/data/Ultradisp.csv')
        
if __name__=='__main__':
    disp = Dispersion()
    disp.pre_data_MLE()
    print(disp.basemean)
