import json
import argparse
from truthfinder import truthfinder

def read_responses_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_problem_type(file_path, problem_type, maling_type):
    n_questions = {"global": 346, "ptsd": 211, "bond-court": 42}.get(problem_type, 0)
    model_path = f"models/{problem_type}/random_order_full_model_0.joblib.gz"
    classifier_path = f"classifiers/{problem_type}/{maling_type}-classifier-{n_questions}.pkl"
    roc_path = f"classifiers/{problem_type}/{maling_type}-roc-{n_questions}.pkl"

    patients_responses = read_responses_from_file(file_path)
    result = truthfinder(patients_responses, model_path, classifier_path, roc_path)
    return result

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Process patient responses based on problem type and maling type.')
parser.add_argument('-file_path', type=str, help='Path to the JSON file containing patient responses.')
parser.add_argument('-problem_type', type=str, choices=['global', 'ptsd', 'bond-court'], help='Problem type (global, ptsd, cchhs).')
parser.add_argument('-maling_type', type=str, help='Maling type.')

args = parser.parse_args()

# Process the problem type based on the input
result = process_problem_type(args.file_path, args.problem_type, args.maling_type)
print(f"{args.problem_type.upper()} results:\n", result)

