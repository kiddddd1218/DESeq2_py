from cProfile import label
from unittest.result import failfast
import Estimate_Size_Factor as ESF
from EstimateDispersionWithPrior import *
from EstimatieDispersion import *
from multiprocessing import Pool, cpu_count
import time
from scipy.stats import norm
import warnings
import plotly.graph_objects as go
class Dispersion:
    def __init__(self,raw_count, designmatrix, project_name) -> None:
        self.raw_count = raw_count
        self.designmatrix = np.array(designmatrix)
        self.normalized_count = None
        self.size_factors = None
        self.logalpha = None
        self.variance_d = None
        self.s_lr = None
        self.length = None
        self.basemean = None
        self.argsl = None
        self.gwdis = None
        self.map = None
        self.final_res = None
        self.num = cpu_count()
        self.gwdispNZ = None
        self.project_name = project_name
        self.disp_res = None
        self.raw_countNZ = None
        self.normalized_countNZ = None

    def pre_data_MLE(self):
        normalized_count, size_factors =  ESF.estimateSizeFactor(self.raw_count)
        self.normalized_count, self.size_factors = normalized_count,size_factors
        self.basemean = self.normalized_count.mean(axis=1)
        self.length = self.raw_count.shape[0]-1
        self.argsl = [(int(i*(self.length//self.num)),int((i+1)*(self.length//self.num))) for i in range(self.num-1)]
        self.argsl.append((int((self.num-1)*(self.length//self.num)) , int(self.length+1)))

    def pre_data_MAP(self):
        raw_countC = self.raw_count
        raw_countC['sum'] = raw_countC.sum(axis=1)
        self.gwdispNZ = pd.to_numeric(self.gwdis[raw_countC['sum'] != 0])
        basemeanNZ = self.basemean[raw_countC['sum'] != 0]
        self.normalized_countNZ = self.normalized_count[raw_countC['sum'] != 0]
        self.raw_countNZ = self.raw_count[raw_countC['sum'] != 0]
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
        pool = Pool(self.num)
        res_MLE = pool.starmap_async(self.func_MLE,self.argsl)
        self.gwdis = pd.concat(res_MLE.get())[0]

    def MAP(self):
        self.pre_data_MAP()
        pool = Pool(self.num)
        res_MAP = pool.starmap_async(self.func_MAP,self.argsl)
        self.map = pd.concat(res_MAP.get())


    def main(self):
        warnings.simplefilter('ignore')
        self.MLE()
        self.MAP()
        difference = np.log(self.gwdispNZ) - self.logalpha
        final_res = self.map.loc[difference.index].squeeze()
        final_res = np.where(difference.squeeze() >= 2*self.s_lr,self.gwdispNZ.squeeze(),final_res)
        final_res = pd.DataFrame(index=difference.index,data=final_res)
        ultra_res = pd.DataFrame(index=self.raw_count.index,data=final_res)
        output_filename = self.project_name+'_dispersion.csv'
        r = ultra_res.to_csv(output_filename)
        self.disp_res = ultra_res
    
    def DispPlot(self):
        fig = go.Figure()
        Mu = self.normalized_countNZ.mean(axis=1)
        sort_arg = np.argsort(Mu)
        MLE = ((self.gwdispNZ).to_numpy())[sort_arg]

        raw_countC = self.raw_count
        raw_countC['sum'] = raw_countC.sum(axis=1)
        final = self.disp_res[raw_countC['sum']!=0].to_numpy()[sort_arg]
        trend = np.exp(self.logalpha)[sort_arg]
        fig.add_trace(go.Scatter(x=Mu,y=MLE,marker='marker',name='gene_est'))
        fig.add_trace(go.Scatter(x=Mu,y=final,marker='marker',name='final'))
        fig.add_trace(go.Scatter(x=Mu,y=trend,marker='line',name='fit'))
        fig.update_xaxes(title_text='mean of normalized counts')
        fig.update_yaxes(title_text='dispersion')
        fig.show()
        
if __name__=='__main__':
    pass
#     disp = Dispersion()
#     disp.main()
