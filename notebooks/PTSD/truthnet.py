import numpy as np
import pandas as pd
import pqdm
from cognet.cognet import cognet
from cognet.cognet import mp_compute, qsampled_distance, distfunc_line, distfunc_multiples
from matplotlib import pyplot as plt
from scipy import stats

class truthnet:
    """
    """
    def __init__(self):
        return NULL
    
    #TODO copied from cognet, extrapolate out of cognet?
    def random_sample(self,
                      samples,
                      type="prob",
                      n=1,
                      steps=200,
                      n_jobs=3):
        """compute a random sample from the underlying distributions of the dataset, by column.
        
        
        Args:
            type (str): How to randomly draw samples. Can take on "null", "uniform", or "prob". Deafults to "prob".
            df (pandas.DataFrame): Desired data to take random sample of. Defaults to None, in which case qnet samples are used.
            n (int): number of random samples to take. Defaults to 1.
            steps (int): number of steps to qsample. Defaults to 1000
            
        Returns:
            return_df (pd.DataFrame): Drawn random sample.
        """
        

        return_df = pd.DataFrame()
        # take random sample from each of the columns based on their probability distribution
        samples = pd.DataFrame(samples)
        if type == "prob":
            for col in samples.columns:
                return_df[col] = samples[col].sample(n=n, replace=True).values
                
        # random sampling using Qnet qsampling
        elif type == "null":
            null_array = np.zeros((len(samples.columns),), dtype=str)
            args = [[null_array, steps] for i in range(n)]
            qsamples = pqdm(args, self.qsampling, n_jobs=n_jobs, argument_type='args') 
            
            # for i in range(n):
            #     qsamples.append(self.qsampling(null_array, steps))
            return_df = pd.DataFrame(qsamples, columns=samples.columns)
            
        # random sampling using uniform distribution of values by Columns
        elif type == "uniform":
            for col in samples.columns:
                # get unqiue values for each column and draw n values randomly
                values = samples[col].unique().astype(str)
                return_df[col]=np.random.choice(values, size=n, replace=True)
        else:
            raise ValueError("Type is not supported!")
        return return_df
    
    def dissonance_histogram(self, dissonance_df, 
                           title="Dissonance of Qnet Samples", 
                           xlabel="Dissonance", 
                           ylabel="Frequency",
                           savefile="dissonance_hist.png"):
        """Bar plot the mean dissonance of each sample

        Args:
            dissonance_df (2d array): dissonance of each element in the sample
            title (String, optional): Title of bar plot. Defaults to None.
            xlabel (String, optional): x-axis label. Defaults to None.
            ylabel (String, optional): y-axis label. Defaults to None.
        """
        pd.DataFrame(dissonance_df).mean(axis=1).hist()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(savefile)
        plt.show()
    
    def cognet_random_reconstruction(self, 
                                     cognet_obj, 
                                     steps=None, 
                                     nsamples=None,
                                     savefile="cognet_randomMaskRecon_results.csv",
                                     save_samples=False):
        """_summary_

        Args:
            cognet_obj (_type_): _description_
            steps (_type_, optional): _description_. Defaults to None.
            nsamples (_type_, optional): _description_. Defaults to None.
            savefile (str, optional): _description_. Defaults to "cognet_randomMaskRecon_results.csv".
            save_samples (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if steps is not None:
            cognet_obj.steps = steps
        if nsamples is not None:
            cognet_obj.set_nsamples(nsamples)
        return cognet_obj.randomMaskReconstruction_multiple(savefile, 
                                                            save_samples=save_samples)
        
    def random_recon_error(self,
                           cognet_obj, 
                           steps=500, 
                           nsamples=None,
                           savefile="cognet_randomMaskRecon_results.csv",
                           indicators=None,
                           plot=True):
        """_summary_

        Args:
            cognet_obj (_type_): _description_
            steps (int, optional): _description_. Defaults to 500.
            nsamples (_type_, optional): _description_. Defaults to None.
            savefile (str, optional): _description_. Defaults to "cognet_randomMaskRecon_results.csv".
            save_samples (bool, optional): _description_. Defaults to False.
        """
        qnet_randommask_df = self.cognet_random_reconstruction(cognet_obj, steps, nsamples, savefile, True)
        
        # original samples
        samples=[]
        for s in qnet_randommask_df['sample']:
            samples.append(list(s))
        qnet_randommask_samples=pd.DataFrame(data=samples, columns=cognet_obj.features, dtype='int').astype(int)

        # reconstructed samples
        qsamples=[]
        for s in qnet_randommask_df['qsampled']:
            qsamples.append(list(s))
        qnet_randommask_qsamples=pd.DataFrame(data=qsamples, columns=cognet_obj.features, dtype='int').replace('',0).astype(int)
        
        # calculate the recon error
        diff_df = qnet_randommask_samples - qnet_randommask_qsamples
        diff_df1 = diff_df.copy()
        diff_df["diff sum"] = diff_df.sum(axis=1)
        num_masked = pd.DataFrame([len(list(s)) for s in qnet_randommask_df['mask_']], columns=["num masked"])
        diff_df["num masked"] = num_masked
        diff_df["recon_results"] = diff_df["diff sum"] / diff_df["num masked"]
        
        # get unique indicators and print average reconstruction results
        if indicators is not None:
            unique_indicators = set(indicators)
            unique_indicators = (list(unique_indicators))
            indicators = pd.DataFrame(indicators)
            for indicator in unique_indicators:
                print("average difference between actual and reconstructed for indicator {}:")
                print(diff_df1[indicators == indicator].mean().mean())
                print("---------------------------------------------------------------------")
        if plot:
            self.recon_err_plot(diff_df1, savefile)
        print("average difference between actual and reconstructed: " + str(diff_df1.mean().mean()))
        return diff_df1
    
    def recon_error_plot(self,
                        df,
                        savefile):
        """_summary_

        Args:
            df (_type_): _description_
            savefile (_type_): _description_
        """
        plt.figure(figsize=[20,6])
        plt.subplots_adjust(hspace=.5)
        df.mean().plot(kind='bar',color='r')
        ax=plt.gca()
        [label.set_visible(False) for x in [ax] for label in x.xaxis.get_ticklabels()[::2]]
        ax.set_title('mean reconstruction error',fontsize=20,fontweight='bold')
        ax.legend(['positive'],fontsize=18)
        ax.legend(['negative'],fontsize=18)
        plt.savefig(savefile[:-3] + "png")
        
    def dissonance_ci(self,
                      cognet_obj,
                      samples,
                      alpha,
                      type,
                      num_samples,
                      savefile="dissonance_matrix.csv"
                      n_jobs=3):
        """_summary_

        Args:
            alpha (_type_): _description_
        """
        # find confidence interval for random samples drawn from uniform distributions
        usamples = self.random_samples(samples, type=type,n=num_samples, n_jobs=n_jobs)
        results = []
        for s in range(len(usamples)):
            results.append(cognet_obj.dissonance(0, sample=usamples.iloc[s]))
        urandom_dissonance_df = pd.DataFrame(results)
        urandom_dissonance_df
        qnet_mean = urandom_dissonance_df.mean(axis=1).mean()
        qnet_std = urandom_dissonance_df.mean(axis=1).std(ddof=1)
        alpha_p1 = 0.1
        alpha_p05 = 0.05
        n_sided = 1 # 1-sided test
        z_crit = stats.norm.ppf(1-alpha_p1/n_sided)
        threshold_p1=(-z_crit*qnet_std)+qnet_mean

        z_crit = stats.norm.ppf(1-alpha_p05/n_sided)
        threshold_p05=(-z_crit*qnet_std)+qnet_mean

        print('Uniform Random Sampling Threshold (90%): ',threshold_p1)
        print('Uniform Random Sampling Threshold (95%): ',threshold_p05)

        plt.figure()
        qnet_dissonance_df = cognet_obj.dissonance_matrix(outfile=savefile, processes=2)
        udissonance_df = pd.DataFrame(data=qnet_dissonance_df.mean(axis=1), columns=["Qnet"])
        udissonance_df["random"] = urandom_dissonance_df.mean(axis=1)
        plt.hist(udissonance_df["Qnet"], alpha=0.5, label="Qnet samples")
        plt.hist(udissonance_df["random"], alpha=0.5, label="random samples")
        plt.legend()
        plt.axvline(threshold_p05, color="red", linestyle="--", alpha=.8)
        plt.show()

    
print(cognet)