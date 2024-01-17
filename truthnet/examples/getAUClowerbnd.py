from truthnet.util import validate
import pandas as pd

resf=pd.read_csv('../../data/res_exp_model2.csv',index_col=0).T
resf.columns=['lower_threshold','veritas','score']
validate(resf,C0=1,C1=0.76,C2=1.35,DX=False,score=True,
         outfile='exptvalid.png',
         validation_type='fnrexpt')