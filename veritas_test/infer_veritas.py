from quasinet.qnet import load_qnet, save_qnet
from quasinet.qnet import qdistance
from quasinet.qsampling import qsample
from quasinet.qnet import membership_degree
import pandas as pd
import numpy as np
from tqdm import tqdm
from quasinet.qnet import Qnet
import dill as pickle  # Dill functions similarly to pickle
import gzip
import shap
from scipy.stats import t, lognorm
from truthfinder import reveal, funcm
from distfit import distfit
from sklearn import metrics
from zedstat import zedstat

# VeRITAS
# objewctive here is to train the veritas model which will be used to determine if a
# subject is being adversarial in a strutured interview. Example of such
# adversarial responses is when a subject is malingering in a
# mental health diagnosis interview or autimated computed aided diatgnostic test
#
#datapath is the path to the survet or interveiew database where we have
# a reasonable number of responses logged from which we will learn our model
# typically we would have a "target label" columns which identifies the ground truth
# on if the subject belonged to one ctaeofry or teh other (have a certain mental health condition, as
# determined by a psychiatrist, or not). Note this column  in not ground truth on malingering, there is typically no ground truth available on that

# index_present describes if the traininmg datafraem/dataset has an index column in teh first column
# training fraction is teh fraction used as training data to learn the "cross-talk or Q-net" models. The reminaing or "test data" is used to infer the three decision thresholds
# query limit is the number of features or items (columns is the data) that are used to dtermine the
# malingering status in deployment. The decision-thresholds are dteremined using this many features, where the features are ordered from most predictive to least predictive using a SHAP analysis

# There are three decision thresholds that the VeRITAS model
#
# QNETmodel
# \Phi(x_{-i}) is a probability distribution of outcomes over the ith variable or question asked, given all other responses (notation for which is x_{-i}. In general we can also have othetr enties missing, and such missing data is interpreted as a distribution over all possible outcomes at that index of missing data. This qnet model allows us to define a metrix between two response vectors x, y denoted as \theta(x,y), and allows us to define the probability Pr(x \rightarrow x) 
#
# LOWER_DECISION THRESHOLD is an estimate of the negative loglikelihood -log Pr(x \rightsrrow x) for a given x per item with a non-missing response. Turns out that as we hav ethis estimate fall below 1, it becomes extreemy unkikley to be naturally generated.
#
# VERITAS THRESHOLD: catures what is teh average deviation of a response vector from what teh model says the responses should be.
#
# UPPER THRESHOLD, estimates a threshold on the ration of loglikelihhods of a response being produced by aqnet inferred fro the positive cases vs that inferred for negative cases
# So for non-malingering response, one needs to be above UPPER threshold, below veritas threshold, and above the LOWER threshold.

class truthnet:
    """
    The truthnet class is designed to train the Veritas model which is used to determine if a
    subject is being adversarial in a structured interview. It is particularly focused on identifying
    adversarial responses in contexts like mental health diagnosis interviews 
    or automated computer-aided diagnostic tests.
    
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
                 target_label,
                 problem='',
                 index_present=True,
                 target_label_positive=1,
                 target_label_negative=0,
                 training_fraction=0.3,
                 threshold_alpha=0.1,
                 threshold_alpha_veritas=1-0.015,
                 query_limit=None,
                 VERBOSE=False,
                 shap_index=None):
        self.datapath = datapath
        self.target_label = target_label
        self.index_present = index_present
        self.target_label_positive = target_label_positive
        self.target_label_negative = target_label_negative
        self.training_fraction = training_fraction
        self.query_limit = query_limit
        if query_limit is None:
            self.query_limit = -1
        self.shap_index = shap_index
        self.data = None  # Placeholder for the data once loaded
        self.veritas_model={}
        self.data = None
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
        Fits the Veritas model to the provided data. It involves training Q-net models
        for both positive and negative cases and determining feature importance using SHAP values.

        Parameters:
            - alpha (float): The significance level for Qnet.
            - shap_index (list): Predefined list of SHAP indices if available.
            - shapnum (int): Number of samples to calculate SHAP values for.
            - nullsteps (int): Number of steps for q-sampling in the background distribution.
            - veritas_version (str): Version identifier for the model.
        """
        
        if self.index_present:
            self.data = pd.read_csv(self.datapath,index_col=0, dtype=str, na_filter=False).fillna('').astype(str)
        else:
            self.data = pd.read_csv(self.datapath, dtype=str, na_filter=False).fillna('').astype(str)


        self.data = self.synccols(self.data)
        
        if self.VERBOSE:
            print(self.data)
        
        num_training=np.rint(self.training_fraction*self.data.index.size).astype(int)

        
        training_index=np.random.choice(self.data.index.values,num_training, replace=False)

        self.training_index=training_index
        df_training=self.data.loc[training_index,:]
        df_training = self.synccols(df_training)
        df_test = self.data.loc[[x for x in self.data.index.values
                                 if x not in training_index],:][df_training.columns]

        if self.target_label:
        
            df_training_pos=df_training[df_training[self.target_label]==str(self.target_label_positive)]
            df_training_neg=df_training[df_training[self.target_label]==str(self.target_label_negative)]
            Xpos_training=df_training_pos.drop(self.target_label,
                                               axis=1)\
                                         .values.astype(str)
            Xneg_training=df_training_neg.drop(self.target_label,
                                               axis=1)\
                                         .values.astype(str)

            featurenames = df_training_pos.drop(self.target_label,
                                                axis=1).columns
            modelneg=Qnet(feature_names=featurenames,alpha=alpha)
            modelneg.fit(Xneg_training)
            modelpos=Qnet(feature_names=featurenames,alpha=alpha)
            modelpos.fit(Xpos_training)
            modelneg.training_index=training_index
            modelpos.training_index=training_index
            
        else:
            featurenames = df_training.columns
            X_training=df_training.values.astype(str)

            model=Qnet(feature_names=featurenames,alpha=alpha)
            model.fit(X_training)
            model.training_index=training_index

        
        def funcw_(S):
            return np.array([membership_degree(s,modelneg)
                             /membership_degree(s,modelpos) for s in S])

        def funcm_(S):
            return funcm(S,model)

        if self.target_label:
            X=df_test.drop(self.target_label,
                           axis=1).values.astype(str)
        
            NULLSTR=np.array(['']*len(modelneg.feature_names))
            s_background=qsample(NULLSTR,modelneg,steps=nullsteps)
            explainer = shap.KernelExplainer(funcw_,np.array([s_background]))
            shap_values = explainer.shap_values(X[:shapnum])
            
            self.shap_index=pd.DataFrame(shap_values.mean(axis=0),
                                         columns=['shap'])\
                              .sort_values('shap',
                                           ascending=False).index.values
            
            modelneg.shap_index=self.shap_index
            modelpos.shap_index=self.shap_index

            # save veritas model
            self.veritas_model['version']=veritas_version
            self.veritas_model['model']=modelpos
            self.veritas_model['model_neg']=modelneg
            self.veritas_model['problem']=self.problem

        else:

            X=df_test.values.astype(str)
        
            NULLSTR=np.array(['']*len(model.feature_names))
            s_background=qsample(NULLSTR,model,steps=nullsteps)
            explainer = shap.KernelExplainer(funcm_,np.array([s_background]))
            shap_values = explainer.shap_values(X[:shapnum])
            
            self.shap_index=pd.DataFrame(shap_values.mean(axis=0),
                                         columns=['shap'])\
                              .sort_values('shap',
                                           ascending=False).index.values
            
            model.shap_index=self.shap_index

            self.veritas_model['version']=veritas_version
            self.veritas_model['model']=model
            self.veritas_model['problem']=self.problem
        
        return

    
    def calibrate(self,
                  qsteps=1000,
                  calibration_num=10000):

        """
        Calibrates the decision thresholds for the Veritas model using the distribution of scores from the trained model. It involves sampling, revealing, and fitting distributions to determine appropriate thresholds.

        Parameters:
            - qsteps (int): Number of steps for q-sampling during calibration.
            - calibration_num (int): Number of samples to use for calibration.
        """

        featurenames=self.veritas_model['model'].feature_names
        NULLSTR=np.array(['']*len(featurenames))
        adict=[]
        for i in tqdm(range(calibration_num)):
            sq=qsample(NULLSTR,self.veritas_model['model'],steps=qsteps)
            ff=pd.DataFrame(sq.reshape(1, -1),
                            columns=featurenames)[featurenames[self.shap_index[:self.query_limit]]]
            adict=np.append(adict,{'xx'+str(i):ff.iloc[0].to_dict()})
        
        Rjson=reveal(list(adict),
                     self.veritas_model,
                     perturb=0,
                     score=(self.target_label is not None),
                     ci=False,
                     model_path=False)
        
        r_=[(x.get('veritas'),
             x.get('score'),
             x.get('lower_threshold')) for x in Rjson[0]]
        r=pd.DataFrame(r_,columns=['veritas','upper','lower'])

        dfit=distfit(distr='lognorm')
        dfit.fit_transform(r.lower.values)
        df,loc,scale=dfit.model['params']
        dist = lognorm(df, loc=loc, scale=scale)
        
        self.veritas_model['dist_lower'] = dist        
        self.veritas_model['LOWER_THRESHOLD'] = dist.ppf(self.threshold_alpha)

        dfitv=distfit(smooth=10,distr='lognorm')
        dfitv.fit_transform(r.veritas.values)
        dfv,locv,scalev=dfitv.model['params']
        distv = lognorm(dfv, loc=locv, scale=scalev)
         
        self.veritas_model['dist_veritas'] = distv        
        self.veritas_model['VERITAS_THRESHOLD'] = distv.ppf(self.threshold_alpha_veritas)

        rn=None
        # use test data to infer the decision_threshold
        if self.target_label:
            df_test = self.data.loc[[x for x in self.data.index.values
                                   if x not in self.training_index],:][np.array(
                                           list(self.veritas_model['model']\
                                                .feature_names)+[self.target_label])]
            featurenames=df_test.drop(self.target_label, axis=1).columns

            df_test = df_test[df_test[self.target_label]!= '']
            df_test[self.target_label] =  df_test[self.target_label].astype(int)

            adictn=df_test[featurenames[self.shap_index[:self.query_limit]]].T.to_dict()
            adictn=[{key:value} for key,value in adictn.items()]

            keyindex=[list(x.keys())[0] for x in adictn]
            labels = df_test.loc[keyindex,self.target_label].values

            Rjsonn=reveal(adictn,
                         self.veritas_model,
                         perturb=0,
                         ci=False,
                         model_path=False)

            rn_=[(x.get('veritas'),
                 x.get('score'),
                 x.get('lower_threshold')) for x in Rjsonn[0]]
            rn=pd.DataFrame(rn_,columns=['veritas','upper','lower'])

            self.veritas_model['calibration_df'] = rn

            pred=rn.upper.values

            fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
            rf=pd.DataFrame(tpr,fpr,columns=['tpr']).assign(threshold=thresholds)
            rf.index.name='fpr'

            self.veritas_model['calibration_roc'] = rf


            zt=zedstat.processRoc(df=rf.reset_index(),
                                  order=3, 
                                  total_samples=2*calibration_num,
                                  positive_samples=calibration_num,
                                  alpha=0.01,
                                  prevalence=0.5)
            zt.smooth(STEP=0.001)
            zt.allmeasures(interpolate=True)
            zt.usample(precision=3)
            Z=zt.get()

            self.veritas_model['upper_scoretoprobability'] =   zt.scoretoprobability
            self.veritas_model['UPPER_THRESHOLD'] = Z[Z.ppv>0.85].threshold.values[-1]
            self.veritas_model['AUC'] = zt.auc()
        
        return r,rn


    def synccols(self, df_):
        df=df_.copy()
        if self.target_label:
            df1 = df[df[self.target_label] ==  str(self.target_label_positive)]
            df0 = df[df[self.target_label] ==  str(self.target_label_negative)]
            col1 = df1.replace('', pd.NA).dropna(axis=1, how='all').columns
            col0 = df0.replace('', pd.NA).dropna(axis=1, how='all').columns
            col = [x for x in col0 if x in col1]
            return df[col]
        else:
            return remove_identical_columns(df_)
    

def save_veritas_model(model, filepath):
    '''
    save veritas model
    '''
    with gzip.open(filepath, 'wb') as file:
        pickle.dump(model, filepath)

        
def load_veritas_model(filepath):
    '''
    load veritas model
    '''
    with gzip.open(filepath, 'rb') as file:
        model = pickle.load(filepath)

    return model


def remove_identical_columns(df):
    # Identify columns where all values are the same
    columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    
    # Drop these columns
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned
