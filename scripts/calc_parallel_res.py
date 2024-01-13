#!/usr/bin/python
import argparse
from truthnet import *
from truthfinder import *
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Calculate response dataframe from response json in parallel")
parser.add_argument("--responsedictpath", type=str, required=True, help="Path to the response dictionary")
parser.add_argument("--Veritas_model_path", type=str, required=True, help="Path to the Veritas model")
parser.add_argument("--outfilepath", type=str, default="res_exp_model2.csv", help="Output file path")
parser.add_argument("--numworkers", type=int, default="10", help="number of workers")

args = parser.parse_args()

responsedictpath = args.responsedictpath
Veritas_model_path = args.Veritas_model_path
outfilepath = args.outfilepath

with gzip.open(responsedictpath, 'rb') as filepath:
    responsedata = pickle.load(filepath)

vmodel=load_veritas_model(Veritas_model_path)
model=vmodel['model']
model_neg=vmodel['model_neg']
perturb=0
H={}
  
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

patients_responses = make_str_format(responsedata)
list_response_dict = extract_ptsd_items(patients_responses)

with ThreadPoolExecutor(max_workers=numworkers) as executor:
    future_to_item = {executor.submit(process_item, i): i for i in list_response_dict}

    for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
        item = future_to_item[future]
        try:
            subjectid, result = future.result()
            H[subjectid] = result
        except Exception as exc:
            print(f'Item {item} generated an exception: {exc}')

hf=pd.DataFrame(H)
hf.to_csv(outfilepath)
print(vmodel) 
