from quasinet.qsampling import qsample
from scipy.stats import entropy
import multiprocessing as mp

import numpy as np
import pandas as pd
from pqdm.threads import pqdm  

class cognet:
    """Aggregate related Qnet functions
    """

    def __init__(self):
        """Init
        """
        self.year = None
        self.n_jobs = 28
        self.qnet = None
        self.steps = 120
        self.num_qsamples = None
        self.all_samples = None
        self.samples = None
        self.samples_as_strings = None
        self.features = []
        self.cols = []
        self.immutable_vars = []
        self.mutable_vars = []
        self.poles = None
        self.polar_features = None
        self.polar_indices = None
        self.poles_dict = {}
        self.d0 = None
        self.s_null = None
        self.D_null = None
        self.mask_prob = 0.5
        self.variation_weight = None
        self.polar_matrix = None
        self.nsamples = None
        self.restricted = False
        self.MAX_PROCESSES = 0
    
    def load_from_model(self,
                        model,
                        data_obj,
                        key,
                        verbose=False):
        """load parameters from model object

        Args:
          model (Class): model obj for loading parameters
          data_obj (class): instance of dataformatter class
          key (str): 'all', 'train', or 'test', corresponding to sample type
          im_vars (list[str], optional): Not implemented yet. Defaults to None.
          m_vars (list[str], optional): Not implemented yet. Defaults to None.
          verbos (bool, optional): Whether or not to print out model state. Defaults to False. 
        """
        if model is not None:
            # inherit atrributes from model object
            self.qnet = model.myQnet
            featurenames, samples = data_obj.format_samples(key)
            samples = pd.DataFrame(samples)
            self.cols = np.array(featurenames)
            self.features = pd.DataFrame(columns=np.array(featurenames))
            
            # inherit mutable and immutable variables from model obj
            if any(x is not None for x in [model.immutable_vars, model.mutable_vars]):
                if model.immutable_vars is not None:
                    self.immutable_vars = model.immutable_vars
                    self.mutable_vars = [x for x in self.features if x not in self.immutable_vars]
                elif model.mutable_vars is not None:
                    self.mutable_vars = model.mutable_vars
                    self.immutable_vars = [x for x in self.features if x not in self.mutable_vars]
            else:
                self.mutable_vars = self.features
            
            # inherit and set class attributes.
            self.samples = pd.DataFrame(samples).replace("nan","").fillna("")
            self.samples.columns = np.array(featurenames)
            self.all_samples = self.samples
            self.samples_as_strings = self.samples.fillna('').values.astype(str)[:]
            self.s_null=['']*len(self.samples_as_strings[0])
            self.D_null=self.qnet.predict_distributions(self.s_null)
            variation_weight = []
            for d in self.D_null:
                v=[]
                for val in d.values():
                    v=np.append(v,val)
                variation_weight.append(entropy(v,base=len(v)))
            variation_weight = np.nan_to_num(variation_weight) # remove nans
            self.variation_weight = variation_weight
        if verbose:
            print("total features: " + str(len(self.features.columns)) + "\n" 
                  + "mutable features: " + str(len(self.mutable_vars.columns)) + "\n"
                  + "immutable features: " + str(len(self.immutable_vars)))
        
    def set_nsamples(self,
                    num_samples,
                    random=False,
                    verbose=True):
        '''select a subset of the samples

        Args:
          num_samples (int): Set num of samples to subset, default to None, resets to all samples
          random (bool): take random sample if true, ordered sample if false. Defaults to False
          verbose (bool): whether or not to print out model state regarding samples. Defaults to True.
        '''
        # each time function is called, reset samples to use_all_samples
        # this allows us to call nsamples numerous times 
        self.samples = self.all_samples
        if self.samples is not None:
            # if a greater number of sample is selected than available, raise error
            if all(x is not None for x in [num_samples, self.samples]):
                if num_samples > len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is greater than the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    raise ValueError(string)

                # if the same number of samples is selected as available, print warning
                if num_samples == len(self.samples.index):
                    string = 'The number of selected samples ({}) ' + \
                        'is equal to the number of samples ({})!'
                    string = string.format(num_samples, len(self.samples.index))
                    print(string)
                    
                # if random is true, return random sample, otherwise return an ordered slice
                if random:
                    self.samples = self.samples.sample(num_samples)
                else:
                    self.samples = self.samples.iloc[:num_samples]
                self.nsamples = num_samples
                self.samples_as_strings = self.samples[self.cols].fillna('').values.astype(str)[:]
                if verbose:
                    if random:
                        print("The number of random samples have been set to " + str(num_samples))
                    else:
                        print("The number of samples have been set to " + str(num_samples))
                
            elif self.samples is None:
                raise ValueError("load_data first!")
        return self.samples
    
    def getBaseFrequency(self, 
                        sample):
        '''get frequency of the variables
        helper func for qsampling

        Args:
          sample (list[str]): vector of sample, must have the same num of features as the qnet
        '''
        # if variable is not mutable, set its base frequency to zero 
        MUTABLE=pd.DataFrame(np.zeros(len(self.cols)),index=self.cols).transpose()
             
        for m in self.mutable_vars:
            MUTABLE[m]=1.0
        mutable_x=MUTABLE.values[0]
        base_frequency=mutable_x/mutable_x.sum()
        
        # otherwise, set base frequency weighted by variation weight
        for i in range(len(base_frequency)):
            if base_frequency[i]>0.0:
                base_frequency[i]= self.variation_weight[i]*base_frequency[i]

        return base_frequency/base_frequency.sum()
    
    def qsampling(self,
                sample,
                steps,
                immutable=False):
        '''perturb the sample based on the qnet distributions and number of steps

        Args:
          sample (1d array-like): sample vector, must have the same num of features as the qnet
          steps (int): number of steps to qsample
          immutable (bool): are there variables that are immutable?
        '''
        # immutable, check that mutable variables have been initialized
        if immutable == True:
            if all(x is not None for x in [self.mutable_vars, sample]):
                return qsample(sample,self.qnet,steps,self.getBaseFrequency(self.samples))
            elif self.mutable_vars is None:
                raise ValueError("set mutable and immutable variables first!")
        else:
            return qsample(sample,self.qnet,steps)

    def random_sample(self,
                      type="prob",
                      df=None,
                      n=1,
                      steps=200,
                      n_jobs=3):
        '''compute a random sample from the underlying distributions of the dataset, by column.
        
        
        Args:
          type (str): How to randomly draw samples. Can take on "null", "uniform", or "prob". Deafults to "prob".
          df (pandas.DataFrame): Desired data to take random sample of. Defaults to None, in which case qnet samples are used.
          n (int): number of random samples to take. Defaults to 1.
          steps (int): number of steps to qsample. Defaults to 1000
          
        Returns:
          return_df (pd.DataFrame): Drawn random sample.
        '''
        # check if a new dataset was given
        if df is None:
            samples_ = self.samples
        else:
            samples_ = df

        return_df = pd.DataFrame()
        # take random sample from each of the columns based on their probability distribution
        if type == "prob":
            for col in samples_.columns:
                return_df[col] = samples_[col].sample(n=n, replace=True).values
                
        # random sampling using Qnet qsampling
        elif type == "null":
            null_array = np.array(['']*len(samples_.columns)).astype('U100')
            args = [[null_array, steps] for i in range(n)]
            qsamples = pqdm(args, self.qsampling, n_jobs=n_jobs, argument_type='args') 
            
            # for i in range(n):
            #     qsamples.append(self.qsampling(null_array, steps))
            return_df = pd.DataFrame(qsamples, columns=samples_.columns)
            
        # random sampling using uniform distribution of values by Columns
        elif type == "uniform":
            for col in samples_.columns:
                # get unqiue values for each column and draw n values randomly
                values = samples_[col].unique().astype(str)
                return_df[col]=np.random.choice(values, size=n, replace=True)
        else:
            raise ValueError("Type is not supported!")
        return return_df

    def dissonance(self,
                    sample_index=0,
                    return_dict=None,
                    MISSING_VAL=0.0,
                    sample=None):
        '''compute dissonance for a single sample, helper function for all_dissonance
        
        Args:
          sample_index (int): index of the sample to compute dissonance. Defaults to 0.
          return_dict (dict): dictionary containing multiprocessing results
          MISSING_VAL (float): default dissonance value
          sample (1D array): sample to compute dissonance of, instead of using sample index. Defaults to None.
          
        Returns: 
          diss[self.polar_indices]: ndarray containing dissonance for sample
        '''
        if all(x is not None for x in [self.samples, self.features]):
            if sample is None:
                s = self.samples_as_strings[sample_index]
            else:
                s = sample
            if self.polar_indices is None:
                self.polar_indices = range(len(s))

            # init vars and calculate dissonance for sample
            Ds=self.qnet.predict_distributions(s)
            diss=np.ones(len(Ds))*MISSING_VAL
            for i in self.polar_indices:
                if s[i] != '':
                    if s[i] in Ds[i].keys():
                        diss[i]=1-Ds[i][s[i]]/np.max(
                            list(Ds[i].values())) 
                    else:
                        diss[i]=1.0
            if return_dict is not None:
                return_dict[sample_index] = diss[self.polar_indices]
            return diss[self.polar_indices]
        else:
            raise ValueError("load_data first!")
    
    def dissonance_matrix(self,
                        outfile='/example_results/DISSONANCE_matrix.csv',
                        processes=6):
        '''get the dissonance for all samples

        Args:
          output_file (str): directory and/or file for output
          processes (int): max number of processes. Defaults to 6.

        Returns:
          result: pandas.DataFrame containing dissonances for each sample
        '''
        # set columns
        if self.polar_indices is not None:
            polar_features = pd.concat([self.features, self.poles], axis=0)
            cols = polar_features[self.cols].dropna(axis=1).columns
        else:
            cols = self.cols
        
        result = mp_compute(self.samples,
                            processes,
                            self.dissonance,
                            cols,
                            outfile)
        return result

def mp_compute(samples,
               max_processes,
               func, 
               cols,
               outfile, 
               args=[]):
    """
    Compute desired function through multiprocessing and save result to csv.

    Args:
        samples (2d array): 2 dimensional numpy array
        processes (int): number of processes to use.
        func (func): function to compute using multiprocessing
        cols (list): column names of resulting csv
        outfile (str)): filepath + filename for resulting csv
        args (list): list containing arguments for desired function. Defaults to empty list.
    """

    # init mp.Manager and result dict
    manager = mp.Manager()
    return_dict = manager.dict()

    num_processes = 0
    process_list = []
    
    # init mp.Processes for each individual sample
    # run once collected processes hit max
    for i in range(len(samples)):
        params = tuple([i] + args + [return_dict])
        num_processes += 1
        p = mp.Process(target=func,
                    args=params)
        process_list.append(p)
        if num_processes == max_processes:
            [x.start() for x in process_list]
            [x.join() for x in process_list]
            process_list = []
            num_processes = 0
            
    # compute remaining processes
    if num_processes != 0:
        [x.start() for x in process_list]
        [x.join() for x in process_list]
        process_list = []
        num_processes = 0
    
    # format and save resulting dict
    result = pd.DataFrame(return_dict.values(), columns=cols, index=return_dict.keys()).sort_index()
    result.to_csv(outfile, index=None)
    return result
