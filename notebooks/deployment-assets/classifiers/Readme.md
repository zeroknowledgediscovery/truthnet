# Classifier details

- All classifiers were trained on dissonances for all available questions. Dissonances were computed from the full model for the data (i.e., using all available samples in the data)
- Classifiers whose name contains `expert` were trained to detect expert malingering (samples drawn from the unconditional distributions of the Dx-positive model, if available, or the full model, otherwise)
- Classifiers whose name contains `runif` were trained to detect random uniform malingering (samples drawn uniformly randomly from the response space of each question)
