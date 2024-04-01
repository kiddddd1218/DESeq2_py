from multiprocessing import Pool, cpu_count
import time

class multi:
    def __init__(self,length,func):
        self.l = length
        self.n = cpu_count()
        self.p = Pool()
        self.f = func
        self.arg = self.cal_arg()

    def cal_arg(self):
        argsl = [(int(i*(self.l//self.n)),int((i+1)*(self.l//self.n))) for i in range(self.n-1)]
        argsl.append((int((self.n-1)*(self.l//self.n)) , int(self.l+1)))
        return argsl
    
    def run(self):
        t1=time.time()
        res = self.p.starmap_async(self.f,self.arg)
        result = res.get()
        t2=time.time()
        print('done in {}'.format(t2-t1))
        return result