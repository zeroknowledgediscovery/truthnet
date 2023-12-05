def getrocprob(raw_scores, rocdata):
    sample_prevalence = rocdata["positive_samples"] / rocdata["total_samples"]
    prevalence = 0.15  # rocdata.get('prevalence',sample_prevalence)

    from zedstat.zedstat import score_to_probability

    prob = score_to_probability(
        raw_scores,
        df=rocdata["roc"],
        total_samples=rocdata["total_samples"],
        positive_samples=rocdata["positive_samples"],
        prevalence=prevalence,
    )
    return prob


def _diss_linear(s, qnet, missing_response="", missing_diss_value=0):
    import numpy as np

    Ds = qnet.predict_distributions(s)
    diss_values = []
    for i in range(len(s)):
        prob = float(Ds[i].get(str(s[i]), np.max(list(Ds[i].values()))))
        max_prob = max(Ds[i].values())
        diss_value = 1 - prob / max_prob if max_prob != 0 else 0
        diss_values.append(diss_value)
    return diss_values


def truthfinder(patients_responses, model_path, classifier_path, roc_path):
    import pandas as pd
    from quasinet.qnet import load_qnet
    import pickle

    patients_responses = [
        {
            patient_id: {
                question_id: str(int(response))
                for question_id, response in patient_responses.items()
            }
            for patient_id, patient_responses in patient.items()
        }
        for patient in patients_responses
    ]

    with open(roc_path, "rb") as file:
        rocdata = pickle.load(file)

    model = load_qnet(model_path)
    classifier = pd.read_pickle(classifier_path)
    all_data_samples = []

    for patient_response in patients_responses:
        for patient_id, responses in patient_response.items():
            resp_df = pd.DataFrame([responses], columns=model.feature_names)
            data_samples = (
                resp_df.fillna("")  # Replace missing values with empty strings
                .astype(str)  # Convert all values to strings
                .values
            )
            all_data_samples.append(data_samples[0])

    diss_values = [_diss_linear(sample, model) for sample in all_data_samples]
    proba = classifier.predict_proba(diss_values)[:, 1]

    estimated_probability_of_event = getrocprob(proba, rocdata)
    output_dict = {}
    for idx, patient_response in enumerate(patients_responses):
        for patient_id in patient_response.keys():
            output_dict[patient_id] = {
                "probability": estimated_probability_of_event[idx][0],
                "ci": tuple(estimated_probability_of_event[idx][1:]),
                "rawscore": proba[idx],
            }
    return output_dict
