def getrocprob(raw_scores, rocdata):
    sample_prevalence = rocdata["positive_samples"] / rocdata["total_samples"]
    prevalence =  rocdata.get('prevalence',sample_prevalence)

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

 
 
def check_json_format(json_data):
    """
    Checks if the given JSON data follows the specified format:
    - JSON is a list
    - Each item in the list is an object with a string or int key (patient ID)
    - Each value corresponding to a patient ID is an object with string or int keys (question IDs) and integer values (answers)
    """
    # Check if the data is a list
    if not isinstance(json_data, list):
        return False

    for item in json_data:
        # Each item in the list should be a dictionary (object)
        if not isinstance(item, dict):
            return False

        for patient_id, questions in item.items():
            # Patient ID should be a string or int
            if not isinstance(patient_id, (str, int)):
                return False

            # Questions should be a dictionary (object)
            if not isinstance(questions, dict):
                return False

            for question_id, answer in questions.items():
                # Question ID should be a string or int
                if not isinstance(question_id, (str, int)):
                    return False

                # Answer should be an integer
                if not isinstance(answer, int):
                    return False

    print('ckeck passed')
    return True
    
    

def make_str_format(resp_json):
    # check format
    if check_json_format(resp_json):
        return  [
            {
                patient_id: {
                    question_id: str(int(response))
                    for question_id,
                    response in patient_responses.items()
                }
                for patient_id, patient_responses in patient.items()
            }
            for patient in resp_json
        ]



def reveal(resp_json, model_path, classifier_path, roc_path):
    import pandas as pd
    from quasinet.qnet import load_qnet
    import dill as pickle

    patients_responses = make_str_format(resp_json)
    
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
