import os
import contextlib
import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
import multiprocessing as mp
from pqdm.threads import pqdm

from quasinet.qsampling import qsample
from dataFormatter import dataFormatter
from model import model


class truthnet:
    """ """

    def __init__(self, qsteps=200, processes=11):
        self.models = {}
        self.data_obj = {}
        self.features = {}
        self.samples = {}
        self.modelpaths = None
        self.MAX_PROCESSES = processes
        self.cithreshold = {}
        self.datapaths = None
        self.commondatapaths = None
        self.suspects = pd.DataFrame()
        self.dissonance = None
        self.QSTEPS = qsteps
        self.suspects = None
        self.coresamples = None
        self.null_dissonance_df = None
        self.null_dissonance_steps = None
        self.year = None
        self.n_jobs = 28
        self.qnet = None
        self.steps = 120
        self.num_qsamples = None
        self.all_samples = None
        self.samples_as_strings = None
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

        return

    def load_data(self, datapaths=None):
        import os

        if datapaths is not None:
            self.datapaths = datapaths

        df = pd.read_csv(self.datapaths[list(self.datapaths)[0]])
        non_null_cols = np.where(df.isna().sum() > -1)[0]
        for model in datapaths:
            df = pd.read_csv(self.datapaths[model])
            non_null_cols = np.intersect1d(
                np.where(df.isna().sum() < len(df))[0], non_null_cols
            )
        for model in datapaths:
            df = pd.read_csv(self.datapaths[model])
            filename = f"{os.path.splitext(self.datapaths[model])[0]}_non-null-cols.csv"
            df.iloc[:, non_null_cols].to_csv(filename, index=False)
            self.datapaths[model] = filename

        for model in datapaths:
            self.data_obj[model] = dataFormatter(samples=self.datapaths[model])
            self.features[model], self.samples[model] = self.data_obj[
                model
            ].Qnet_formatter()
        self.length = len(self.features[list(self.features)[0]])
        return self.features, self.samples

    def fit(self, fit=True, processes=None, modelpaths=None):
        from model import model

        if modelpaths is not None:
            self.modelpaths = modelpaths

        if processes is None:
            processes = self.MAX_PROCESSES

        for mod in modelpaths:
            self.models[mod] = model()

        if fit:
            for mod in modelpaths:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    data_ = dataFormatter(samples=self.datapaths[mod])
                    model_ = model()
                    model_.fit(
                        data_obj=data_,
                        min_samples_split=2,
                        alpha=0.05,
                        max_depth=-1,
                        max_feats=-1,
                        early_stopping=False,
                        verbose=0,
                        random_state=None,
                        njobs=processes,
                    )
                    model_.save(self.modelpaths[mod], low_mem=True)

        for model in modelpaths:
            self.models[model].load(self.modelpaths[model])
            # self.load_from_model(self.models[model], self.data_obj[model], "all")
        return

    def getDissonance(self, processes=11, outfile=None):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.set_nsamples(len(self.samples), random=False, verbose=False)
            self.MAX_PROCESSES = processes
            return_dict = self.dissonance_matrix(
                outfile=outfile, processes=self.MAX_PROCESSES
            )
        self.dissonance = pd.DataFrame(return_dict.copy())
        return

    def generateRandomResponse(
        self,
        n=1,
        processes=None,
        samples=None,
        mode="prob",
        alpha=0.05,
        n_sided=1,
        getsuspects=True,
        steps=200,
    ):
        """random sample from the underlying distributions by column.

        Args:
            type (str): Mode, can be "null", "uniform", or "prob" (Default)
            df (pandas.DataFrame): Data. If None, qnet samples are used.
            n (int): number of random samples to take. Defaults to 1.
            steps (int): number of steps to qsample. Defaults to 1000

        Returns:

        """
        if processes is None:
            processes = self.MAX_PROCESSES
        usamples = self.random_sample(
            df=samples, type=mode, n=n, steps=steps, n_jobs=processes
        )
        results = []
        for s in range(len(usamples)):
            results.append(self.compute_dissonance(0, sample=usamples.iloc[s]))
        if getsuspects:
            self.urandom_dissonance_df = pd.DataFrame(results)
        else:
            self.null_dissonance_df = pd.DataFrame(results)
            self.null_dissonance_steps = steps

        if getsuspects:
            self.__cithreshold(alpha=alpha, n_sided=n_sided, mode="suspect")
        else:
            self.__cithreshold(alpha=alpha, n_sided=n_sided, mode="core")

        return usamples

    def __cithreshold(self, alpha, n_sided=1, mode="suspect"):
        if mode == "suspect":
            qnet_mean = self.urandom_dissonance_df.mean(axis=1).mean()
            qnet_std = self.urandom_dissonance_df.mean(axis=1).std(ddof=1)
        if mode == "core":
            qnet_mean = self.null_dissonance_df.mean(axis=1).mean()
            qnet_std = self.null_dissonance_df.mean(axis=1).std(ddof=1)

        z_critL = norm.ppf(1 - alpha / n_sided)
        z_critR = norm.ppf(1 - (1 - alpha) / n_sided)
        self.cithreshold[(mode, alpha)] = (
            (-z_critL * qnet_std) + qnet_mean,
            (-z_critR * qnet_std) + qnet_mean,
        )
        return

    def getSuspects(
        self,
        samples=None,
        alpha=0.05,
        mode="uniform",
        return_samples=False,
        processes=None,
    ):
        """get suspects at a specified significance level using trained Qnet.

        Args:
            samples (int): No. of random samples drawn (no. of samples used to construct QNet)
            alpha (float): significance level (default=0.05).
            processes (int): max number of parallel processes used (default=10).

        Returns:
            suspects (pandas.DataFrame)
        """

        self.suspects = None
        if samples is None:
            samples = len(self.samples)
        if processes is None:
            processes = self.MAX_PROCESSES

        usamples = self.generateRandomResponse(
            n=samples, processes=processes, steps=None, mode=mode, alpha=alpha
        )

        if self.dissonance is not None:
            mean_dissonance = pd.DataFrame(
                data=self.dissonance.mean(axis=1), columns=["mean_dissonance"]
            )
        else:
            raise ("dissonance values are None")

        self.suspects = mean_dissonance[
            mean_dissonance.mean_dissonance >= self.cithreshold[("suspect", alpha)][0]
        ].copy()

        self.suspects.drop_duplicates(inplace=True)
        if return_samples:
            return self.suspects.copy(), usamples
        else:
            return self.suspects.copy()

    def getCoresamples(
        self, samples=None, alpha=0.05, steps=None, return_samples=False, processes=None
    ):
        """get samples in model-core at a specified significance level using trained Qnet.

        Args:
            samples (int): No. of random samples drawn (no. of samples used to construct QNet)
            alpha (float): significance level (default=0.05).
            processes (int): max number of parallel processes used (default=10).

        Returns:
            suspects (pandas.DataFrame)
        """
        # alpha=1-alpha
        self.suspects = None
        if samples is None:
            samples = len(self.samples)
        if processes is None:
            processes = self.MAX_PROCESSES
        if steps is None:
            steps = self.QSTEPS

        usamples = self.generateRandomResponse(
            n=samples,
            processes=processes,
            steps=steps,
            mode="null",
            getsuspects=False,
            alpha=alpha,
        )

        results = []
        for s in range(len(usamples)):
            results.append(self.compute_dissonance(0, sample=usamples.iloc[s]))
        if self.dissonance is not None:
            mean_dissonance = pd.DataFrame(
                data=self.dissonance.mean(axis=1), columns=["mean_dissonance"]
            )
        else:
            raise ("dissonance values are None")

        self.coresamples = mean_dissonance[
            mean_dissonance.mean_dissonance <= self.cithreshold[("core", alpha)][1]
        ].copy()

        self.coresamples.drop_duplicates(inplace=True)
        if return_samples:
            return self.coresamples.copy(), usamples
        else:
            return self.coresamples.copy()

    def load_from_model(self, model, data_obj, key, verbose=False):
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
            self.qnet = model.qnet
            featurenames, samples = data_obj.format_samples(key)
            samples = pd.DataFrame(samples)
            self.cols = np.array(featurenames)
            self.features = pd.DataFrame(columns=np.array(featurenames))

            # inherit mutable and immutable variables from model obj
            if any(x is not None for x in [model.immutable_vars, model.mutable_vars]):
                if model.immutable_vars is not None:
                    self.immutable_vars = model.immutable_vars
                    self.mutable_vars = [
                        x for x in self.features if x not in self.immutable_vars
                    ]
                elif model.mutable_vars is not None:
                    self.mutable_vars = model.mutable_vars
                    self.immutable_vars = [
                        x for x in self.features if x not in self.mutable_vars
                    ]
            else:
                self.mutable_vars = self.features

            # inherit and set class attributes.
            self.samples = pd.DataFrame(samples).replace("nan", "").fillna("")
            self.samples.columns = np.array(featurenames)
            self.all_samples = self.samples
            self.samples_as_strings = self.samples.fillna("").values.astype(str)[:]
            self.s_null = [""] * len(self.samples_as_strings[0])
            self.D_null = self.qnet.predict_distributions(self.s_null)
            variation_weight = []
            for d in self.D_null:
                v = []
                for val in d.values():
                    v = np.append(v, val)
                variation_weight.append(entropy(v, base=len(v)))
            variation_weight = np.nan_to_num(variation_weight)  # remove nans
            self.variation_weight = variation_weight
        if verbose:
            print(
                "total features: "
                + str(len(self.features.columns))
                + "\n"
                + "mutable features: "
                + str(len(self.mutable_vars.columns))
                + "\n"
                + "immutable features: "
                + str(len(self.immutable_vars))
            )

    def set_nsamples(self, num_samples, random=False, verbose=True):
        """select a subset of the samples

        Args:
          num_samples (int): Set num of samples to subset, default to None, resets to all samples
          random (bool): take random sample if true, ordered sample if false. Defaults to False
          verbose (bool): whether or not to print out model state regarding samples. Defaults to True.
        """
        # each time function is called, reset samples to use_all_samples
        # this allows us to call nsamples numerous times
        self.samples = self.all_samples
        if self.samples is not None:
            # if a greater number of sample is selected than available, raise error
            if all(x is not None for x in [num_samples, self.samples]):
                if num_samples > len(self.samples.index):
                    string = (
                        "The number of selected samples ({}) "
                        + "is greater than the number of samples ({})!"
                    )
                    string = string.format(num_samples, len(self.samples.index))
                    raise ValueError(string)

                # if the same number of samples is selected as available, print warning
                if num_samples == len(self.samples.index):
                    string = (
                        "The number of selected samples ({}) "
                        + "is equal to the number of samples ({})!"
                    )
                    string = string.format(num_samples, len(self.samples.index))
                    print(string)

                # if random is true, return random sample, otherwise return an ordered slice
                if random:
                    self.samples = self.samples.sample(num_samples)
                else:
                    self.samples = self.samples.iloc[:num_samples]
                self.nsamples = num_samples
                self.samples_as_strings = (
                    self.samples[self.cols].fillna("").values.astype(str)[:]
                )
                if verbose:
                    if random:
                        print(
                            "The number of random samples have been set to "
                            + str(num_samples)
                        )
                    else:
                        print(
                            "The number of samples have been set to " + str(num_samples)
                        )

            elif self.samples is None:
                raise ValueError("load_data first!")
        return self.samples

    def getBaseFrequency(self, sample):
        """get frequency of the variables
        helper func for qsampling

        Args:
          sample (list[str]): vector of sample, must have the same num of features as the qnet
        """
        # if variable is not mutable, set its base frequency to zero
        MUTABLE = pd.DataFrame(np.zeros(len(self.cols)), index=self.cols).transpose()

        for m in self.mutable_vars:
            MUTABLE[m] = 1.0
        mutable_x = MUTABLE.values[0]
        base_frequency = mutable_x / mutable_x.sum()

        # otherwise, set base frequency weighted by variation weight
        for i in range(len(base_frequency)):
            if base_frequency[i] > 0.0:
                base_frequency[i] = self.variation_weight[i] * base_frequency[i]

        return base_frequency / base_frequency.sum()

    def qsampling(self, sample, steps, immutable=False):
        """perturb the sample based on the qnet distributions and number of steps

        Args:
          sample (1d array-like): sample vector, must have the same num of features as the qnet
          steps (int): number of steps to qsample
          immutable (bool): are there variables that are immutable?
        """
        # immutable, check that mutable variables have been initialized
        if immutable:
            if all(x is not None for x in [self.mutable_vars, sample]):
                return qsample(
                    sample, self.qnet, steps, self.getBaseFrequency(self.samples)
                )
            elif self.mutable_vars is None:
                raise ValueError("set mutable and immutable variables first!")
        else:
            return qsample(sample, self.qnet, steps)

    def random_sample(self, type="prob", df=None, n=1, steps=200, n_jobs=3):
        """compute a random sample from the underlying distributions of the dataset, by column.


        Args:
          type (str): How to randomly draw samples. Can take on "null", "uniform", or "prob". Deafults to "prob".
          df (pandas.DataFrame): Desired data to take random sample of. Defaults to None, in which case qnet samples are used.
          n (int): number of random samples to take. Defaults to 1.
          steps (int): number of steps to qsample. Defaults to 1000

        Returns:
          return_df (pd.DataFrame): Drawn random sample.
        """
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
            null_array = np.array([""] * len(samples_.columns)).astype("U100")
            args = [[null_array, steps] for i in range(n)]
            qsamples = pqdm(args, self.qsampling, n_jobs=n_jobs, argument_type="args")

            # for i in range(n):
            #     qsamples.append(self.qsampling(null_array, steps))
            return_df = pd.DataFrame(qsamples, columns=samples_.columns)

        # random sampling using uniform distribution of values by Columns
        elif type == "uniform":
            for col in samples_.columns:
                # get unqiue values for each column and draw n values randomly
                values = samples_[col].unique().astype(str)
                return_df[col] = np.random.choice(values, size=n, replace=True)
        else:
            raise ValueError("Type is not supported!")
        return return_df

    def compute_dissonance(
        self, sample_index=0, return_dict=None, MISSING_VAL=0.0, sample=None
    ):
        """compute dissonance for a single sample, helper function for all_dissonance

        Args:
          sample_index (int): index of the sample to compute dissonance. Defaults to 0.
          return_dict (dict): dictionary containing multiprocessing results
          MISSING_VAL (float): default dissonance value
          sample (1D array): sample to compute dissonance of, instead of using sample index. Defaults to None.

        Returns:
          diss[self.polar_indices]: ndarray containing dissonance for sample
        """
        if all(x is not None for x in [self.samples, self.features]):
            if sample is None:
                s = self.samples_as_strings[sample_index]
            else:
                s = sample
            if self.polar_indices is None:
                self.polar_indices = range(len(s))

            # init vars and calculate dissonance for sample
            Ds = self.qnet.predict_distributions(s)
            diss = np.ones(len(Ds)) * MISSING_VAL
            for i in self.polar_indices:
                if s[i] != "":
                    if s[i] in Ds[i].keys():
                        diss[i] = 1 - Ds[i][s[i]] / np.max(list(Ds[i].values()))
                    else:
                        diss[i] = 1.0
            if return_dict is not None:
                return_dict[sample_index] = diss[self.polar_indices]
            return diss[self.polar_indices]
        else:
            raise ValueError("load_data first!")

    def _diss_linear(s, qnet, missing_value=0):
        diss = list()
        Ds = qnet.predict_distributions(s)

        for i in range(len(s)):
            if s[i] != "":
                if s[i] in Ds[i].keys():
                    diss.append(1 - Ds[i][s[i]] / np.max(list(Ds[i].values())))
                elif s[i] == "missing":
                    diss.append(missing_value)
                else:
                    diss.append(1)

        return np.array(diss)

    def _diss_log(s, qnet, missing_value=0):
        diss = list()
        Ds = qnet.predict_distributions(s)

        for i in range(len(s)):
            if s[i] != "":
                if s[i] in Ds[i].keys():
                    diss.append(-np.log(Ds[i][s[i]]))
                elif s[i] == "missing":
                    diss.append(missing_value)
                else:
                    diss.append(np.inf)
            # else:
            # diss.append(missing_value)

        return np.array(diss)

    def _actual_sample_dissonance(
        self, data_sample, diss_models, diss_fcn, order, length, missing_value=0
    ):
        if order is None:
            order = range(length)

        sample = np.full(length, "", dtype="<U21")

        diss = [list() for model in diss_models]

        for i in order:
            if data_sample[i] == "":
                sample[i] = "missing"
            else:
                sample[i] = data_sample[i]
        # indent to compute diss at each question
        for d, model in zip(diss, diss_models):
            d.append(diss_fcn(sample, model, missing_value))

        return sample, diss

    def _all_actual_samples_dissonance(
        self, data_samples, diss_models, diss_fcn, order, length, missing_value=0
    ):
        samples = list()
        dissonances = list()

        for data_sample in data_samples:
            samp, diss = self._actual_sample_dissonance(
                data_sample, diss_models, diss_fcn, order, length, missing_value
            )
            samples.append(samp)
            dissonances.append(diss)

        return samples, dissonances

    def _sample_with_dissonance(
        self,
        sample_model,
        length,
        diss_models,
        diss_fcn=_diss_linear,
        order=None,
        data_samples=None,
        method=None,
        avg_non_missing=None,
        missing_value=0,
    ):
        from quasinet.utils import sample_from_dict
        import random

        if order is None:
            order = range(length)

        if avg_non_missing is not None:
            order_ind = list(range(length))
            random.shuffle(order_ind)
            order_ind = sorted(order_ind[: int(np.floor(avg_non_missing))])
        else:
            order_ind = order

        if data_samples is not None:
            data_samples_df = pd.DataFrame(data_samples)
            data_sample_values = pd.Series(
                {
                    col: [x for x in data_samples_df[col].unique() if x != ""]
                    for col in data_samples_df
                }
            )

        sample = np.full(length, "")

        diss = [list() for model in diss_models]

        for i in order:
            if sample_model is not None:
                if method == "unconditional":
                    if i in order_ind:
                        prob_dict = sample_model.predict_distribution(
                            np.full(length, ""), i
                        )
                        sample[i] = sample_from_dict(prob_dict)
                    else:
                        sample[i] = "missing"
                else:
                    if i in order_ind:
                        prob_dict = sample_model.predict_distribution(sample, i)
                        sample[i] = sample_from_dict(prob_dict)
                    else:
                        sample[i] = "missing"
            else:
                if i in order_ind:
                    sample[i] = random.choice(data_sample_values[i])
                else:
                    sample[i] = "missing"
        # indent to compute diss at each question
        [
            d.append(diss_fcn(sample, model, missing_value))
            for d, model in zip(diss, diss_models)
        ]

        return sample, diss

    def _nsamples_with_dissonance(
        self,
        n_samples,
        sample_model,
        length,
        diss_models,
        diss_fcn=_diss_linear,
        order=None,
        data_samples=None,
        method=None,
        avg_non_missing=None,
        missing_value=0,
    ):
        samples = list()
        dissonances = list()

        for i in range(n_samples):
            samp, diss = self._sample_with_dissonance(
                sample_model,
                length,
                diss_models,
                diss_fcn,
                order,
                data_samples,
                method,
                avg_non_missing,
                missing_value,
            )
            samples.append(samp)
            dissonances.append(diss)

        return samples, dissonances

    def _dissonance_data_at_question(self, dissonances, questions_asked):
        return np.array(
            [np.hstack([d[questions_asked - 1] for d in diss]) for diss in dissonances]
        )

    # generate samples under the given models and compute dissonances under specified diss_models
    def sampling_scenario(
        self,
        n_m2_samples=None,
        m2_model=None,
        diss_models=None,
        length=None,
        m2_method="unconditional",
        n_runif_samples=None,
        diss_fcn=_diss_linear,
        order=None,
        data_samples=None,
        missing_value=0,
        avg_non_missing=None,
    ):
        samples = {}
        dissonances = {}

        if data_samples is None:
            data_samples = self.samples["full"]

        samples["actual"], dissonances["actual"] = self._all_actual_samples_dissonance(
            data_samples,
            diss_models,
            diss_fcn,
            order,
            length,
            missing_value,
        )

        if n_m2_samples is not None:
            if m2_model is None:
                if "pos" in self.models:  # dx mode
                    m2_model = self.models["pos"].qnet
                else:  # core mode
                    m2_model = self.models["full"].qnet

            samples["m2"], dissonances["m2"] = self._nsamples_with_dissonance(
                n_m2_samples,
                m2_model,
                length,
                diss_models,
                diss_fcn,
                order,
                method=m2_method,
                avg_non_missing=avg_non_missing,
                missing_value=missing_value,
            )

        if n_runif_samples is not None:
            samples["runif"], dissonances["runif"] = self._nsamples_with_dissonance(
                n_runif_samples,
                None,
                length,
                diss_models,
                diss_fcn,
                order,
                data_samples,
                avg_non_missing=avg_non_missing,
                missing_value=missing_value,
            )

        self.generated_samples = samples
        self.generated_dissonances = dissonances

        return samples, dissonances

    def _diss_dataset(self, dissonances, questions_asked=1, groups=["actual", "m2"]):
        diss_dataset = pd.concat(
            [
                pd.DataFrame(
                    self._dissonance_data_at_question(
                        dissonances[group], questions_asked
                    )
                ).assign(y=group)
                for group in groups
            ]
        )
        return diss_dataset

    def train_classifier(
        self,
        groups=["actual", "m2"],
        n_maling_samples=10,
        diss_models=None,
        roc_outfile=None,
        classifier_outfile=None,
        plot=False,
    ):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        from sklearn.metrics import (
            RocCurveDisplay,
            roc_curve,
        )
        from sklearn.model_selection import train_test_split

        if diss_models is None:
            diss_models = [
                self.models[list(self.models)[i]].qnet for i in range(len(self.models))
            ]

        if groups[1] == "m2":
            self.sampling_scenario(
                n_m2_samples=n_maling_samples,
                n_runif_samples=None,
                diss_models=diss_models,
                length=self.length,
            )
        if groups[1] == "runif":
            self.sampling_scenario(
                n_runif_samples=n_maling_samples,
                n_m2_samples=None,
                diss_models=diss_models,
                length=self.length,
            )

        data = self._diss_dataset(self.generated_dissonances, groups=groups)
        # data = data.iloc[:, np.r_[:questions_asked, -1]]
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop("y", axis="columns"), data["y"], test_size=0.33
        )

        classifier = RandomForestClassifier(n_jobs=-1)

        classifier.fit(
            X_train.to_numpy(),
            y_train.replace(
                {
                    groups[0]: 0,
                    groups[1]: 1,
                }
            ),
        )

        if plot is True:
            RocCurveDisplay.from_estimator(
                classifier,
                X_test.to_numpy(),
                y_test.replace(
                    {
                        groups[0]: 0,
                        groups[1]: 1,
                    }
                ),
                pos_label=1,
                name=groups[1],
            )

        fpr, tpr, thresholds = roc_curve(
            y_test.replace(
                {
                    groups[0]: 0,
                    groups[1]: 1,
                }
            ),
            classifier.predict_proba(X_test.to_numpy())[:, 1],
            pos_label=1,
        )
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

        roc_crv = {}
        roc_crv["roc"] = roc_df
        roc_crv["total_samples"] = len(y_train)
        roc_crv["positive_samples"] = len(y_train[y_train == groups[1]])

        if roc_outfile is None:
            self.roc_outfile = f"roc-crv-{groups[1]}.pkl"
        else:
            self.roc_outfile = roc_outfile
        if classifier_outfile is None:
            self.classifier_outfile = f"classifier-{groups[1]}.pkl"
        else:
            self.classifier_outfile = classifier_outfile

        pd.to_pickle(roc_crv, self.roc_outfile)

        pd.to_pickle(
            classifier,
            self.classifier_outfile,
        )

        self.roc_curve = roc_crv
        self.classifier = classifier

        return

    def show_parameters(self):
        print(f"Data paths: {self.datapaths}")
        print(f"Model paths: {self.modelpaths}")
        print(f"Roc path: {self.roc_outfile}")
        print(f"Classifier path: {self.classifier_outfile}")

    def dissonance_matrix(
        self, outfile="/example_results/DISSONANCE_matrix.csv", processes=6
    ):
        """get the dissonance for all samples

        Args:
          output (str): directory and/or file for output
          processes (int): max number of processes. Defaults to 6.

        Returns:
          result: pandas.DataFrame containing dissonances for each sample
        """
        # set columns
        if self.polar_indices is not None:
            polar_features = pd.concat([self.features, self.poles], axis=0)
            cols = polar_features[self.cols].dropna(axis=1).columns
        else:
            cols = self.cols

        result = mp_compute(
            self.samples, processes, self.compute_dissonance, cols, outfile
        )
        return result


def mp_compute(samples, max_processes, func, cols, outfile, args=[]):
    """
    Compute desired function through multiprocessing and save result to csv.

    Args:
        samples (2d array): 2 dimensional numpy array
        max_processes (int): number of processes to use.
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
        p = mp.Process(target=func, args=params)
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

    # format and save resulting dict
    result = pd.DataFrame(
        return_dict.values(), columns=cols, index=return_dict.keys()
    ).sort_index()
    result.to_csv(outfile, index=None)
    return result
