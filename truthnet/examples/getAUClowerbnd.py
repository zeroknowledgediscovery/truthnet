from truthnet.util import validate
import pandas as pd

resf1=pd.read_csv('../../data/res_exp_model2.csv',index_col=0).T
resf1.columns=['lower_threshold','veritas','score']

resf=pd.read_csv('../../data/res_R20_model2.csv',index_col=0).T
resf.columns=['lower_threshold','veritas','score']
resf['dx'] = [int(x[-1]) for x in resf.index.values]

import numpy as np
from tqdm import tqdm
R={}
for c0 in tqdm(np.arange(.8,1.1,.025)):
    b=validate(resf,C0=c0,C1=0.76,C2=1.35,DX=True,
               score=True,plots=False,verbose=False,
               validation_type='withdx')
    
    a=validate(resf1,C0=c0,C1=0.76,C2=1.35,DX=False,score=True,plots=False,
               validation_type='fnrexpt')

    R[c0]=(1-a['fnr'],1-b['mratio'],1 - 0.5*(a['fnr'] + b['mratio']+a['fnr']*b['mratio'] ))

rf=pd.DataFrame(R).T
rf.columns=['sensitivity','maxspecificity','minauc']
print(rf)

rf.index.name='thresholds'
rff=rf[['sensitivity','maxspecificity']].reset_index()
rff=rff.rename(columns={'sensitivity':'tpr'})
rff['fpr']=1-rff.maxspecificity
rff=rff.drop('maxspecificity',axis=1)
print(rff)

from zedstat import zedstat
zt=zedstat.processRoc(df=rff,
                      order=3, 
                      total_samples=304,
                      positive_samples=86,
                      alpha=0.01,
                      prevalence=0.5)
zt.smooth(STEP=0.1)
zt.allmeasures(interpolate=True)
zt.usample(precision=2)
print(zt.auc())

import pylab as plt
rf.plot(x='maxspecificity',y='sensitivity')
plt.show()

