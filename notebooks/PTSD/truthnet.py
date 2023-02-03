import numpy as np
import pandas as pd
import pqdm
from cognet.cognet import cognet
from cognet.cognet import mp_compute, qsampled_distance, distfunc_line, distfunc_multiples
from matplotlib import pyplot as plt
from scipy import stats
from cognet.dataFormatter import dataFormatter
from cognet.cognet import cognet as cg
from quasinet.qnet import qdistance, save_qnet
from cognet.model import model 
import os
import contextlib


class truthnet:
    """
    """
    def __init__(self,qsteps=200,processes=11,datapath=None):
        
        self.cognet_obj = cg()
        self.model_obj = model()
        self.data_obj=None
        self.features=None
        self.samples=None
        self.modelpath=None
        self.cognet_obj.MAX_PROCESSES=processes
        self.cithreshold={}
        self.datapath=None
        self.suspects=pd.DataFrame()
        self.dissonance=None
        self.QSTEPS=qsteps
        self.suspects=None
        self.edgesamples=None
        
        return 
    
    def load_data(self,datapath=None):
        if datapath is not None:
            self.datapath=datapath
        self.data_obj=dataFormatter(samples=self.datapath)
        self.features,self.samples = self.data_obj.Qnet_formatter()
        return self.features,self.samples
    
    def fit(self,fit=True,
            save=True,
            processes=None,
            modelpath=None):
        if modelpath is not None:
            self.modelpath=modelpath

        if processes is None:
            processes=self.cognet_obj.MAX_PROCESSES
        
        if fit:
            self.model_obj.fit(data_obj=data_obj,
                               njobs=processes)
            save_qnet(self.model_obj.myQnet,
                      self.modelpath,
                      low_mem=False)
        else:
            self.model_obj.load(self.modelpath)
        
        self.cognet_obj.load_from_model(self.model_obj,
                                        self.data_obj,
                                        'all')
        return
        
    def getDissonance(self,processes=11,outfile=None):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.cognet_obj.set_nsamples(len(self.samples),
                                         random=False,
                                         verbose=False)
            self.cognet_obj.MAX_PROCESSES = processes
            return_dict = self.cognet_obj.dissonance_matrix(
                outfile=outfile,processes=self.cognet_obj.MAX_PROCESSES)
        self.dissonance = pd.DataFrame(return_dict.copy())
        return
    
    def generateRandomResponse(self,
                               n=1,
                               processes=None,
                               samples=None,
                               mode='prob',
                               alpha=0.05,
                               n_sided=1,
                               steps=200):
        """random sample from the underlying distributions by column.
        
        Args:
            type (str): Mode, can be "null", "uniform", or "prob" (Default)
            df (pandas.DataFrame): Data. If None, qnet samples are used.
            n (int): number of random samples to take. Defaults to 1.
            steps (int): number of steps to qsample. Defaults to 1000
            
        Returns:
            
        """
        if processes is None:
                processes = self.cognet_obj.MAX_PROCESSES
        usamples = self.cognet_obj.random_sample(df=samples,
                                            type=mode,
                                            n=n,
                                            steps=steps,
                                            n_jobs=processes)
        results = []
        for s in range(len(usamples)):
            results.append(
                self.cognet_obj.dissonance(0,
                                           sample=usamples.iloc[s]))
        self.urandom_dissonance_df = pd.DataFrame(results)

        self.__cithreshold(alpha=alpha,n_sided=n_sided)
        return
    
    def __cithreshold(self,
                      alpha,
                      n_sided = 1 ):
        qnet_mean = self.urandom_dissonance_df.mean(axis=1).mean()
        qnet_std = self.urandom_dissonance_df.mean(axis=1).std(ddof=1)
        z_crit = stats.norm.ppf(1-alpha/n_sided)
        self.cithreshold[alpha]=(-z_crit*qnet_std)+qnet_mean
        return 


    def getSuspects(self,
                    samples=None,
                    alpha=0.05,
                    processes=None):
        """get suspects at the specified significance level based on trained Qnet.
        
        Args:
            samples (int): No. of random samples drawn (no. of samples in dataframe used to construct QNet)
            alpha (float): significance level (default=0.05).
            processes (int): max number of parallel processes used (default=10).
            
        Returns:
            suspects (pandas.DataFrame)
        """
        
        self.suspects=None
        if samples is None:
            samples=len(self.samples)
        if processes is None:
            processes=self.cognet_obj.MAX_PROCESSES

        self.generateRandomResponse(n=samples,
                                    processes=processes,
                                    steps=None,
                                    mode='uniform',
                                    alpha=alpha)

        mean_dissonance=pd.DataFrame(
            data=self.dissonance.mean(axis=1), columns=["mean_dissonance"])

        self.suspects=mean_dissonance[mean_dissonance.mean_dissonance
                                 >=self.cithreshold[alpha]].copy()

        self.suspects.drop_duplicates(inplace=True)
        return self.suspects.copy()

    
    def getEdgesamples(self,
                       samples=None,
                       alpha=0.05,
                       steps=None,
                       processes=None):
        """get samples out-of-model-core at the specified significance level based on trained Qnet.
        
        Args:
            samples (int): No. of random samples drawn (no. of samples in dataframe used to construct QNet)
            alpha (float): significance level (default=0.05).
            processes (int): max number of parallel processes used (default=10).
            
        Returns:
            suspects (pandas.DataFrame)
        """
        
        self.suspects=None
        if samples is None:
            samples=len(self.samples)
        if processes is None:
            processes=self.cognet_obj.MAX_PROCESSES
        if steps is None:
            steps=self.QSTEPS                
           

        self.generateRandomResponse(n=samples,
                                    processes=processes,
                                    steps=steps,
                                    mode='null',
                                    alpha=alpha)

        mean_dissonance=pd.DataFrame(
            data=self.dissonance.mean(axis=1), columns=["mean_dissonance"])

        self.edgesamples=mean_dissonance[mean_dissonance.mean_dissonance
                                 >=self.cithreshold[alpha]].copy()

        self.edgesamples.drop_duplicates(inplace=True)
        return self.edgesamples.copy()

    












    
