from quasinet.qnet import Qnet, load_qnet, save_qnet, qdistance, membership_degree
from quasinet.qsampling import qsample
import pandas as pd
import numpy as np
from tqdm import tqdm
import dill as pickle  # Dill functions similarly to pickle for serialization
import gzip
import shap  # SHAP is used for explaining predictions
from scipy.stats import lognorm  # Used for statistical functions
from truthfinder import reveal  # Function from a custom package for truth finding
from distfit import distfit  # Distribution fitting
from sklearn import metrics  # For calculating metrics like AUC
from zedstat import zedstat  # Custom statistical package
from concurrent.futures import ProcessPoolExecutor

class truthnet:
    """
    The truthnet class is designed to train the Veritas model which is used to determine if a
    subject is being adversarial in a structured interview. It is particularly focused on identifying
    adversarial responses in contexts like mental health diagnosis interviews or automated computer-aided diagnostic tests.
    
    Attributes:
        - datapath (str): Path to the survey or interview database with logged responses.
        - target_label (str): Name of the column in the dataset that identifies the ground truth.
        - index_present (bool): Specifies if the first column in the dataset is an index column.
        - target_label_positive (int): Value indicating a positive case for the target condition.
        - target_label_negative (int): Value indicating a negative case for the target condition.
        - training_fraction (float): Fraction of data used as training data to learn the Q-net models.
        - query_limit (int): Number of features used to determine malingering status in deployment.
        - shap_index (list): Ordered list of indices based on SHAP values for feature importance.
        - problem (str): A description or identifier for the type of problem being addressed.
        - threshold_alpha (float): Significance level for lower decision threshold.
        - threshold_alpha_veritas (float): Significance level for Veritas threshold.
    """
    
    def __init__(self, datapath,
                 target_label=None,
                 problem='',
                 index_present=True,
                 target_label_positive=1,
                 target_label_negative=0,
                 training_fraction=0.3,
                 threshold_alpha=0.1,
                 threshold_alpha_veritas=1-0.015,
                 query_limit=None,
                 shap_index=None,
                 VERBOSE=False):
        """
        Initializes the truthnet class with the provided parameters and sets up the structure for the Veritas model.
        """
        self.datapath = datapath
        self.target_label = target_label
        self.index_present = index_present
        self.target_label_positive = target_label_positive
        self.target_label_negative = target_label_negative
        self.training_fraction = training_fraction
        self.query_limit = query_limit if query_limit is not None else -1
        self.shap_index = shap_index
        self.data = None  # Placeholder for data once loaded
        self.veritas_model = {}
        self.problem = problem
        self.training_index = None
        self.threshold_alpha = threshold_alpha
        self.threshold_alpha_veritas = threshold_alpha_veritas
        self.VERBOSE = VERBOSE
        
    def fit(self,
            alpha=0.1,
            shap_index=None,
            shapnum=10,
            nullsteps=100000,
            veritas_version='0.0.1'):
        """
        Fits the Veritas/Qnet model to the provided data. 
        It involves training Q-net models for both positive and negative cases and determining feature importance using SHAP values
        which allows us to pick a good order of posing questions/items.

        Parameters:
            - alpha (float): The significance level for Qnet.
            - shap_index (list): Predefined list of SHAP indices if available.
            - shapnum (int): Number of samples to calculate SHAP values for.
            - nullsteps (int): Number of steps for q-sampling in the background distribution.
            - veritas_version (str): Version identifier for the model.
        """
        # Data loading and preparation
        self.data = pd.read_csv(self.datapath, index_col=0) if self.index_present else pd.read_csv(self.datapath)
        num_training = np.rint(self.training_fraction * self.data.index.size).astype(int)
        training_index = np.random.choice(self.data.index.values, num_training, replace=False)
        self.training_index = training_index
        df_training = self.data.loc[training_index, :]
        df_test = self.data.loc[[x for x in self.data.index.values if x not in training_index], :]
        
        # Splitting data into positive and negative cases for training
        df_training_pos = df_training[df_training[self.target_label] == self.target_label_positive]
        df_training_neg = df_training[df_training[self.target_label] == self.target_label_negative]
        Xpos_training = df_training_pos.drop(self.target_label, axis=1, errors='ignore').values.astype(str)
        Xneg_training = df_training_neg.drop(self.target_label, axis=1, errors='ignore').values.astype(str)
        featurenames = df_training_pos.drop(self.target_label, axis=1, errors='ignore').columns
        
        # Training Q-net models for positive and negative cases
        modelneg = Qnet(feature_names=featurenames, alpha=alpha)
        modelneg.fit(Xneg_training)
        modelpos = Qnet(feature_names=featurenames, alpha=alpha)
        modelpos.fit(Xpos_training)
        modelneg.training_index = training_index
        modelpos.training_index = training_index
        
        # SHAP analysis for feature importance
        def funcw_(S):
            return np.array([membership_degree(s, modelneg) / membership_degree(s, modelpos) for s in S])
        
        X = df_test.drop(self.target_label, axis=1, errors='ignore').values.astype(str)
        NULLSTR = np.array([''] * len(modelneg.feature_names))
        s_background = qsample(NULLSTR, modelneg, steps=nullsteps)
        explainer = shap.KernelExplainer(funcw_, np.array([s_background]))
        shap_values = explainer.shap_values(X[:shapnum])
        self.shap_index = pd.DataFrame(shap_values.mean(axis=0), columns=['shap']).sort_values('shap', ascending=False).index.values
        
        # Saving trained models and SHAP indices in the Veritas model dictionary
        modelneg.shap_index = self.shap_index
        modelpos.shap_index = self.shap_index
        self.veritas_model['version'] = veritas_version
        self.veritas_model['model'] = modelpos
        self.veritas_model['model_neg'] = modelneg
        self.veritas_model['problem'] = self.problem
        
        return

    
    def calibrate(self, qsteps=1000, calibration_num=10000):
        """
        Calibrates the decision thresholds for the Veritas model.
 
        Parameters:
            - qsteps (int): Number of steps for q-sampling during calibration.
            - calibration_num (int): Number of samples to use for calibration.
        """
        # Sampling and calibration setup
        featurenames = self.veritas_model['model'].feature_names
        NULLSTR = np.array([''] * len(featurenames)).astype('U100')

        num_workers=11
        model = self.veritas_model['model']

        if self.VERBOSE:
            print('calibrating,,,')
        
        def task(i):
            s=qsample(NULLSTR, self.veritas_model['model'], steps=qsteps)
            return funcm(s,model),dissonance_distr_median(s,model)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            lv_ = list(tqdm.tqdm(executor.map(task, range(calibration_num)), total=calibration_num))

       lower_ = np.array([x[0] for x in lv_])
       veritas_ = np.array([x[1] for x in lv_])
        


    #
    #    lower_=[funcm(qsample(NULLSTR, self.veritas_model['model'],
    #                         steps=qsteps),model) for i in tqdm(range(calibration_num))]
    #    veritas_=[dissonance_distr_median(qsample(NULLSTR, self.veritas_model['model'],
    #                         steps=qsteps),model) for i in tqdm(range(calibration_num))]

        self.veritas_model['calibration_lower']=lower_
        self.veritas_model['calibration_veritas']=veritas_

        # Fitting distributions to lower and veritas thresholds
        dfit = distfit(distr='lognorm')
        dfit.fit_transform(lower_)
        df, loc, scale = dfit.model['params']
        dist = lognorm(df, loc=loc, scale=scale)
        self.veritas_model['dist_lower'] = dist        
        self.veritas_model['LOWER_THRESHOLD'] = dist.ppf(self.threshold_alpha)

        dfitv = distfit(smooth=10, distr='lognorm')
        dfitv.fit_transform(veritas_)
        dfv, locv, scalev = dfitv.model['params']
        distv = lognorm(dfv, loc=locv, scale=scalev)
        self.veritas_model['dist_veritas'] = distv        
        self.veritas_model['VERITAS_THRESHOLD'] = distv.ppf(self.threshold_alpha_veritas)

        # Using test data to infer the decision threshold for the upper threshold
        if self.target_label:
            df_test = self.data.loc[[x for x in self.data.index.values if x not in self.training_index], :]
            featurenames = df_test.drop(self.target_label, axis=1, errors='ignore').columns
            adictn = df_test[featurenames[self.shap_index[:self.query_limit]]].T.to_dict()
            adictn = [{key: value} for key, value in adictn.items()]
            keyindex = [list(x.keys())[0] for x in adictn]
            labels = df_test.loc[keyindex, self.target_label].values
            
            Rjsonn = reveal(adictn, self.veritas_model, perturb=0, ci=False, model_path=False)
            rn_ = [(x.get('veritas'), x.get('score'), x.get('lower_threshold')) for x in Rjsonn[0]]
            rn = pd.DataFrame(rn_, columns=['veritas', 'upper', 'lower'])
            pred = rn.upper.values
            
            # Calculating metrics and determining the upper threshold
            fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
            rf = pd.DataFrame(tpr, fpr, columns=['tpr']).assign(threshold=thresholds)
            rf.index.name = 'fpr'
            zt = zedstat.processRoc(df=rf.reset_index(), order=3, total_samples=2*calibration_num,
                                    positive_samples=calibration_num, alpha=0.01, prevalence=0.5)
            zt.smooth(STEP=0.001)
            zt.allmeasures(interpolate=True)
            zt.usample(precision=3)
            Z = zt.get()
            
            self.veritas_model['upper_scoretoprobability'] = zt.scoretoprobability
            
            if Z.ppv.values[0] > 0.85:
                THR=0.85
            else:
                THR=Z.ppv.values[2]
                
            self.veritas_model['UPPER_THRESHOLD'] = Z[Z.ppv > THR].threshold.values[-1]
            self.veritas_model['AUC'] = zt.auc()
        
        return rn
