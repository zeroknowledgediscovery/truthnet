#!/~/.pyenv/shims/python3.9

import sys
sys.path.append("../") 
from infer_veritas import  *
from truthfinder import *
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

#with gzip.open('../response_jsons/validation_cat.pkl.gz', 'rb') as filepath:
#    catadata = pickle.load(filepath)
with gzip.open('../response_jsons/validation_index20.pkl.gz', 'rb') as filepath:
    fulldata20 = pickle.load(filepath)
#with gzip.open('../response_jsons/validation_ptsd_full.pkl.gz', 'rb') as #filepath:
#    fulldata = pickle.load(filepath)


Veritas_model_path='../veritas_models/veritas_ptsd.pkl.gz'
Veritas_model_path='../veritas_models/veritas_001.pkl.gz'

#import gz
import dill as pickle
with gzip.open(Veritas_model_path, 'rb') as file:
    vmodel=pickle.load(file)

model=vmodel['model']
model_neg=vmodel['model_neg']
perturb=0
H={}

# Function to handle processing each item
def process_item(i):
    subjectid = i['subject_id']
    resp = i['responses']
    s = pd.concat([pd.DataFrame(columns=model.feature_names),
                   pd.DataFrame(resp,index=['response'])])\
            .fillna('').values[0].astype(str)

    veritas = dissonance_distr_median(s, model)
    score = funcw(s, model, model_neg)
    lowert = funcm(s, model)
    
    return subjectid, (lowert, veritas, score)



patients_responses = make_str_format(fulldata20)
list_response_dict = extract_ptsd_items(patients_responses)


# Set up ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit tasks to the executor
    future_to_item = {executor.submit(process_item, i): i for i in list_response_dict}

    # Process as they complete
    for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
        item = future_to_item[future]
        try:
            subjectid, result = future.result()
            H[subjectid] = result
        except Exception as exc:
            print(f'Item {item} generated an exception: {exc}')

hf=pd.DataFrame(H)
hf.to_csv('res1.csv')