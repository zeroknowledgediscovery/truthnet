from truthnet.util import validate
import pandas as pd

resf=pd.read_csv('../../data/res_exp_model2.csv',index_col=0).T
resf.columns=['lower_threshold','veritas','score']
a=validate(resf,C0=1,C1=0.76,C2=1.35,DX=False,score=True,plots=False,
         validation_type='fnrexpt')

resf=pd.read_csv('../../data/res_R20_model2.csv',index_col=0).T
resf.columns=['lower_threshold','veritas','score']
resf['dx'] = [int(x[-1]) for x in resf.index.values]
b=validate(resf,C0=1,C1=0.76,C2=1.35,DX=True,score=True,plots=False,
         validation_type='withdx')

print(a,b)
