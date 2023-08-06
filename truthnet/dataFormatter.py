import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import assert_None


class dataFormatter:
    """format data to be suitable for Qnet training and testing
    """

    def __init__(self,
                 samples):
        """init

        Args:
            samples ([str], optional): 2D array with rows as observations and columns as features.
        """
        self.samples = pd.read_csv(samples,keep_default_na=False) # keep_default_na=False added
        self.features = {}
        self.nan_cols = []
        self.immutable_vars = None
        self.mutable_vars = None
        self.test_size = None
        self.random_state = None
        self.train_data = None
        self.test_data = None

    def __train_test_split(self,
                           test_size,
                           train_size=None,
                           random_state=None):
        """split the samples into training and testing samples

        Args:
          test_size (float): fraction of sample to take as test_size.
          train_size (float): fraction of sample to take as train_size. Defaults to None, and 1-test_size
          random_state (int, optional): random seed to split samples dataset . Defaults to None.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.train_data, self.test_data = train_test_split(self.samples,
                                                           test_size=test_size,
                                                           train_size=train_size,
                                                           random_state=random_state)
    
    def Qnet_formatter(self,
                       samples=None,
                       key=None):
        """format data for Qnet input

        Args:
          samples ([str], optional): 2D array with rows as observations and columns as features.
          key (str): Either 'train' or 'test' key, to determine which set of features
        
        Returns:
            features and samples of either the train and test dataset
        """
        # if not isinstance(samples, np.ndarray):
        #     raise ValueError('Samples must be in numpy array form!')
        if samples is None:
            samples = self.samples
        features = np.array(samples.columns.astype(str)[:])
        samples = samples.replace("nan","").fillna("").values.astype(str)[:]
        # remove columns that are all NaNs
        not_all_nan_cols = ~np.all(samples == '', axis=0)
        self.nan_cols = np.all(samples == '', axis=0)

        samples = samples[:, not_all_nan_cols]
        
        features = features[not_all_nan_cols]
        features = list(features)
        if key is not None:
            self.features[key] = features
        return features, samples

    def format_samples(self,
                       key,
                       test_size=.5):
        """formats samples and featurenames, either all, train, or test
        
        Args:
          key (str): 'all', 'train', or 'test', corresponding to sample type

        Returns:
            samples and featurenames: formatted
        """
        
        
        if all(x is None for x in [self.train_data,
                                       self.test_data,
                                       self.samples]):
            raise ValueError("Split samples into test and train datasets or input samples first!")
        if key == 'train':
            self.__train_test_split(1-test_size)
            samples = self.train_data
        elif key == 'test':
            self.__train_test_split(test_size)
            samples = self.test_data
        elif key == 'all':
            samples = self.samples
        else:
            raise ValueError("Invalid key, key must be either 'all', 'test', or 'train")
        
        return self.Qnet_formatter(samples, key=key)
    