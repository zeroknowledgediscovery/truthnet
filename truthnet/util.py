import argparse
from .truthnet import *
from .truthfinder import *
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
import pylab as plt
import seaborn as sns
from sklearn import metrics
from zedstat import zedstat

def calculate_response_parallel(responsedictpath,
                                Veritas_model_path,
                                outfilepath,
                                verbose=False,
                                numworkers=10):

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
    if verbose:
        print(vmodel) 


def get_malinger_func(C0,C1,C2,score=True):
    '''
    -1 is malingering
    0 is no dx
    1 is true dx
    '''
    if score:
        def malinger(row):
            if (row.lower_threshold < C0) and (row.score > C2):
                return -1
            if (row.veritas > C1) and (row.score > C2):
                return -1    
            if (row.score < C2):
                return 0
            else:
                return 1

    else:
        def malinger(row):
            if (row.lower_threshold < C0):
                return -1
            if (row.veritas > C1) and (row.score > C2):
                return -1    
            if (row.score < C2):
                return 0
            else:
                return 1
            
    return malinger


def validate(response_dataframe,C0,C1,C2,
             DX=True,score=True, plots=True,verbose=True,
             validation_type='withdx',outfile='report.png'):
    '''
    response dataframe should look like:
           lower_threshold veritas score
    sub1         x            x       x
    sub2         x            x       x
    DX=True implies a dx column is present in input
    score=False implies the score is faked or absent (no dx subgroup 
    was available in training
    '''
    malinger=get_malinger_func(C0,C1,C2,score=score)
    if not DX:
        response_dataframe['dx'] = [x>C2 for x in response_dataframe.score.values]

    response_dataframe['mg']=response_dataframe.apply(malinger,axis=1)


    if DX:
        fpr, tpr, thresholds = metrics.roc_curve(response_dataframe.dx.values.astype(int),
                                                 response_dataframe.score.values.astype(float),
                                                 pos_label=1)
        ff=pd.DataFrame(tpr,fpr,columns=['tpr']).assign(threshold=thresholds)
        ff.index.name='fpr'
        zt=zedstat.processRoc(df=ff.reset_index(),
                              order=3, 
                              total_samples=304,
                              positive_samples=86,
                              alpha=0.01,
                              prevalence=0.5)
        zt.smooth(STEP=0.001)
        zt.allmeasures(interpolate=True)
        zt.usample(precision=3)
        Z=zt.get()
        if verbose:
            print(Z[Z.ppv>.875].tail(10))


    if validation_type == "withdx":
        mratio=(response_dataframe[(response_dataframe.mg==-1)
                                   & (response_dataframe.dx==1)].index.size)/response_dataframe.dx.sum()
        fullauc=zt.auc()
        
        if plots:
            #plt.style.use('seaborn-dark-palette')

            plt.figure(figsize=[20,12])
            plt.subplot(231)
            sns.scatterplot(data=response_dataframe,x='lower_threshold',y='veritas',hue='mg',size='dx')
            plt.plot([.5,2.5],[.76,.76],'-r')
            plt.plot([C0,C0],[.5,.95],'-r')

            plt.subplot(232)
            ax=sns.scatterplot(data=response_dataframe,x='score',y='veritas',hue='dx')
            plt.plot([.2,2.5],[C1,C1],'-r')
            plt.plot([C2,C2],[.5,.95],'-r')

            plt.subplot(233)
            sns.scatterplot(data=response_dataframe,x='score',y='lower_threshold',hue='dx')
            plt.subplots_adjust(wspace=0.23)  # Adjust this value as needed

            cf=response_dataframe.corr()
            plt.subplot(234)
            sns.heatmap(cf,cmap='jet',alpha=.5)


            plt.subplot(235)

            plt.plot(fpr,tpr,'g',lw=2)
            plt.gca().legend(['R20'])
            zt.get().tpr.plot(style='-b',lw=2)

            ax = plt.subplot(236)
            ax.text(0.5, 0.6, f'malinger prevalenec in DX: {mratio:.2f}', fontsize=16, ha='center')
            ax.text(0.5, 0.4, f'AUC: {fullauc[0]:.2f} $\pm$ {fullauc[1]-fullauc[0]:.2f}', fontsize=16, ha='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
        return {'auc':fullauc,'mratio':mratio}, response_dataframe, zt

    if validation_type == "fnrexpt":
        fnr=response_dataframe[(response_dataframe.mg==1)].index.size/response_dataframe.index.size
        if plots:
            #plt.style.use('seaborn-dark-palette')

            plt.figure(figsize=[20,12])
            plt.subplot(231)
            sns.scatterplot(data=response_dataframe,x='lower_threshold',y='veritas',hue='mg',size='dx')
            plt.plot([.5,2.5],[.76,.76],'-r')
            plt.plot([C0,C0],[.5,.95],'-r')

            plt.subplot(232)
            ax=sns.scatterplot(data=response_dataframe,x='score',y='veritas',hue='dx')
            plt.plot([.2,2.5],[C1,C1],'-r')
            plt.plot([C2,C2],[.5,.95],'-r')

            plt.subplot(233)
            sns.scatterplot(data=response_dataframe,x='score',y='lower_threshold',hue='dx')
            plt.subplots_adjust(wspace=0.23)  # Adjust this value as needed

            cf=response_dataframe.corr()
            plt.subplot(234)
            sns.heatmap(cf,cmap='jet',alpha=.5)

            ax = plt.subplot(236)
            ax.text(0.5, 0.6, f'FNR in EXPT: {fnr:.2f}', fontsize=16, ha='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        return {'fnr':fnr}, response_dataframe


    if validation_type == "noscore":
        mrate=response_dataframe[response_dataframe.mg==-1].index.size/response_dataframe.index.size
        if plots:
            #plt.style.use('seaborn-dark-palette')

            plt.figure(figsize=[8,8])
            plt.subplot(111)
            sns.scatterplot(data=response_dataframe,x='lower_threshold',y='veritas',hue='mg')
            plt.plot([.1,2.5],[C1,C1],'-r')
            plt.plot([C0,C0],[.1,.95],'-r')
            plt.plot([C2,C2],[.1,.95],'-r')

            ax = plt.gca()

            ax.text(0.65, 0.8, f'mrate: {mrate:.2f}', fontsize=16, ha='center')

        return {'mrate':mrate}, response_dataframe

    plt.savefig(outfile,dpi=300,bbox_inches='tight',transparent=True)   




def drop_empty_string_keys(input_dict):
    # Create a new dictionary, excluding keys with empty string values
    cleaned_dict = {key: value for key, value in input_dict.items() if value != ''}
    return cleaned_dict
