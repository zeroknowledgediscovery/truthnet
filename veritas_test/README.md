#objewctive here is to train the veritas model which will be used to determine if a
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

