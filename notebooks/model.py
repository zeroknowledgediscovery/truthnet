from quasinet.qnet import Qnet, load_qnet, save_qnet
from util import assert_None

from io import BytesIO
import requests

class model:
    """Facilitate training and constructing Qnet
    """

    def __init__(self):
        """Init
        """
        self.myQnet = None
        self.features = None
        self.immutable_vars = None
        self.mutable_vars = None
        self.data_obj = None

    def fit(self,
            featurenames=None,
            samples=None,
            data_obj=None,
            min_samples_split=2,
            alpha=0.05,
            max_depth=-1,
            max_feats=-1,
            early_stopping=False,
            verbose=0,
            random_state=None,
            njobs=4):
        """fit Quasinet Qnet model

        Args:
          featurenames ([str], optional): names of the model features. Defaults to None.
          samples ([str], optional): 2D array with rows as observations and columns as features. Defaults to None.
          data_obj (obj, optional): Build Qnet directly from data obj without other inputs. Defaults to None.
          njobs (int, optional): Number of jobs used to fit Qnet. Defaults to 2.
        """
        num_None = assert_None([featurenames,samples,data_obj], raise_error=False)
        if num_None == 0:
            raise ValueError("input either samples and features or data object, not both!")
        elif data_obj is not None:
            featurenames, samples=data_obj.Qnet_formatter() # returns the training data
            print(len(samples))
            self.immutable_vars, self.mutable_vars = data_obj.immutable_vars, data_obj.mutable_vars
        elif num_None > 1:
            raise ValueError("input both samples and features or data object!")
        print("training Qnet -------------")
        self.myQnet = Qnet(n_jobs=njobs, feature_names=featurenames,
                           min_samples_split=min_samples_split, alpha=alpha,
                           max_depth=max_depth, max_feats=max_feats,
                           early_stopping=early_stopping,
                           verbose=verbose, random_state=random_state)
        self.myQnet.fit(samples)
        print("Qnet trained --------------")
        self.features = featurenames

    def save(self,
             file_path=None,
             low_mem = False):
        """save qnet

        Args:
          file_path (str, optional): Desired Qnet filename. Defaults to None.
        """
        assert_None([self.myQnet])
        if file_path is None:
            file_path = 'tmp_Qnet.joblib'
        save_qnet(self.myQnet, file_path, low_mem=low_mem)
    
    def load(self,
             file_path,
             VERBOSE=False):
        """load Qnet from file

        Args:
          file_path (str): path to Qnet savefile
          VERBOSE (bool): boolean to turn on verbose

        Returns:
          [Qnet]: Qnet object
        """
        if VERBOSE:
            print("loading..")
        # check if reading from url or local directory
        if "https" in file_path or "www." in file_path:
            url_file = BytesIO(requests.get(file_path).content)
            self.myQnet = load_qnet(url_file)
        else:
            self.myQnet = load_qnet(file_path)
        self.features = self.myQnet.feature_names
        if VERBOSE:
            print("done")
        return self.myQnet
