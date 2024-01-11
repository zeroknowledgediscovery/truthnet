import gzip
import dill as pickle
import pandas as pd
from quasinet.qnet import load_qnet
from quasinet.qnet import qdistance
from quasinet.qsampling import qsample
from quasinet.qnet import membership_degree
import numpy as np


DIAGNOSIS_THRESHOLD=1.35
VERITAS_THRESHOLD=0.76
LOWER_FABRICATION_THREHOLD=1


def check_json_format(json_data):
    """
    Checks if the given JSON data follows the specified format:
    - JSON is a list
    - Each item in the list is an object with a string or int key (patient ID)
    - Each value corresponding to a patient ID is an object with string or int keys (question IDs) and integer values (answers)
    """
    # Check if the data is a list
    if not isinstance(json_data, (np.ndarray,list)):
        print('not nd.array or list')
        return False

    for item in json_data:
        # Each item in the list should be a dictionary (object)
        if not isinstance(item, dict):
            print('items not dict')
            return False

        for patient_id, questions in item.items():
            # Patient ID should be a string or int
            if not isinstance(patient_id, (str, int)):
                print('pid not str or int')
                return False

            # Questions should be a dictionary (object)
            if not isinstance(questions, dict):
                print('questions not dict')
                return False

            for question_id, answer in questions.items():
                # Question ID should be a string or int
                if not isinstance(question_id, (str, int)):
                    print('question id not str or int')
                    return False

                # Answer should be an integer
                if not isinstance(answer, (str,int)):
                    print('question response not str or int')
                    return False

    print('ckeck passed')
    return True
    
    
def make_str_format(resp_json):
    # check format
    if check_json_format(resp_json):
        return [
            {
                patient_id: {
                    question_id: str(int(response)) if response != '' else response
                    for question_id, response in patient_responses.items()
                }
                for patient_id, patient_responses in patient.items()
            }
            for patient in resp_json
        ]


def load_from_pkl_gz(filename):
    """
    Unpickles and loads the contents of a .pkl.gz file.

    :param filename: The path to the .pkl.gz file.
    :return: The unpickled data.
    """
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dissonance(pos,seq,model):
    if seq[pos]=='':
        return np.nan
    D=model.predict_distributions(seq) 
    return 1-D[pos].get(str(seq[pos]),0)

def dissonance_distr(seq,model):
    return np.array([dissonance(pos,seq,model) for pos in range(len(seq))])

def dissonance_distr_median(seq,model):
    a=dissonance_distr(seq,model)
    return np.median(a[~np.isnan(a)])


# Function to read the JSON file and extract the top-level keys
# as subject ID and the associated dictionary as responses
def extract_ptsd_items(jsondata):
        data=jsondata
        subjects = []
        for entry in data:
            for subject_id, responses in entry.items():
                subjects.append({"subject_id": subject_id,
                                 "responses": responses})
        return subjects

def funcw(s,model_pos,model_neg):
    '''
    funcw should be greater than DIAGNOSIS_THRESHOLD
    '''
    neg=membership_degree(s,model_neg)
    pos=membership_degree(s,model_pos)
    
    return neg/pos

def funcm(array,model_pos,model_neg=None):
    '''
    funcm should be greater than LOWER_FABRICATION_THREHOLD=1. Lower values indicate fabrication
    '''
    if isinstance(array, np.ndarray):
        if isinstance(array[0], str):
            return -membership_degree(array,model_pos)/(array!='').sum() 
        if isinstance(array[0],np.ndarray):
            if isinstance(array[0][0], str):
                return np.array([-membership_degree(s,model_pos)/(s!='').sum()  for s in array])
    raise('incorrect datatype. must be 2d numpy array of strings')
    return


    
def reveal(resp_json,
           veritas_model_path,
           perturb=3,
           score=True,
           ci=True,
           model_path=True):
    patients_responses = make_str_format(resp_json)
    
    list_response_dict = extract_ptsd_items(patients_responses)

    if model_path:
        veritas_model = load_from_pkl_gz(veritas_model_path)
    else:
        veritas_model = veritas_model_path

    message=[]
    for i in list_response_dict:
        subjectid=i['subject_id']
        resp = i['responses']
        s=pd.concat([pd.DataFrame(columns=veritas_model['model'].feature_names),
                   pd.DataFrame(resp,index=['response'])])\
                        .fillna('').values[0].astype(str)

        if perturb > 0:
            s=qsample(s,veritas_model['model'],steps=perturb)
            
        i['veritas'] = dissonance_distr_median(s,veritas_model['model'])

        if score:
            i['score']=funcw(s,
                         veritas_model['model'], 
                         veritas_model['model_neg'])
        else:
            i['score']=0
            
        i['lower_threshold']=funcm(s,
                         veritas_model['model'])
        if ci:
            i['veritas_prob'] = veritas_model['dist_veritas'].cdf(i['veritas'])
            i['lower_prob'] = veritas_model['dist_lower'].cdf(i['lower_threshold'])

        message = message + [interpret(i)]
    return list_response_dict, message


def interpret(calculated_score):
    lower_threshold=calculated_score.get('lower_threshold',None)
    score=calculated_score.get('score',None)
    veritas=calculated_score.get('veritas',None)
    veritas_prob=calculated_score.get('veritas_prob',None)


    MESSAGE={-1:'No PTSD indicated. Malingering test unnecessary',
             0:'No Malingering detected. True PTSD indicated',
             1:'Fabrication detected',
             2:'Maligering detected. You are likely lying with probability > '+ str(veritas_prob)[:5]}

    if lower_threshold < LOWER_FABRICATION_THREHOLD:
        malingering_class=1
    else:
        if score > DIAGNOSIS_THRESHOLD:
            if veritas > VERITAS_THRESHOLD:
                malingering_class=2
            else:
                malingering_class=0
        else:
             malingering_class=-1
                
    return MESSAGE[malingering_class]
