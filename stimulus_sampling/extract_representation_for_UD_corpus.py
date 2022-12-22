import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model, ArtificialSubject, load_dataset
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
import xarray as xr
import argparse
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
UD_PARENT='/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
SAVE_DIR='/om2/user/ehoseini/MyData/brain-score-language/dataset'
OUTPUT_DIR='/om2/user/ehoseini/MyData/brain-score-language/output'
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='packaging first level files for subject')
parser.add_argument('model_id', type=str)
args=parser.parse_args()

if __name__ == '__main__':
    model_id = int(args.model_id)
    save_path = Path(SAVE_DIR, 'ud_sentencez_data_token_filter_v3_brainscore_WO_ANNSet1.pkl')
    if not save_path.exists():
        p=Path(UD_PARENT, 'ud_sentencez_data_token_filter_v3_no_dup.pkl')
        assert p.exists()
        ud_sentences=pd.read_pickle(p.__str__())
        # first cleanup ud_sentences:
        ud_sentence_clean=[]
        for sent_id, sent_ in tqdm(enumerate(ud_sentences)):
            if np.unique([len(x) for k, x in sent_.items() if type(x) == list]).shape[0] == 1:
                ud_sentence_clean.append(sent_)
            else:
                print(f'sentence {sent_id} has inconsistent values, dropping it')
        # drop annset 1
        ud_text=[x['text'] for x in ud_sentence_clean]
        ANNdata=load_dataset('ANNSet1_fMRI.train.language_top_90')
        drop_sent_id=[]
        for sent in ANNdata.sentence:
            drop_sent_id.append(ud_text.index(sent.values))
        clean_WO_ANNSet1_id=set(np.arange(len(ud_sentence_clean)))-set(drop_sent_id)
        ud_sentence_clean=[ud_sentence_clean[x] for x in clean_WO_ANNSet1_id]
        ud_sentences_xr = []
        for sent_id,sent_ in tqdm(enumerate(ud_sentence_clean)):
            True
            sent_pd = pd.DataFrame(sent_)
            sent_pd['stimulus_id']=sent_id
            ud_sentences_xr.append(sent_pd.to_xarray())

        ud_sentences_xr=xr.concat(ud_sentences_xr,dim='index')
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(ud_sentences_xr, f)
    else:
        ud_sentences_xr = pd.read_pickle(save_path)

    models = ['roberta-base', 'xlm-mlm-en-2048', 'xlnet-large-cased', 'albert-xxlarge-v2', 'bert-base-uncased',
              'gpt2-xl', 'ctrl']
    model=models[model_id]
    candidate = load_model(f'{model}')
    candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                     recording_type=ArtificialSubject.RecordingType.fMRI)

    predictions = []
    for stim_id, stim in tqdm(ud_sentences_xr.groupby('stimulus_id'), desc='digest individual sentences'):
        assert len(np.unique(stim.text))==1
        sent_string=np.unique(stim.text)[0]
        #if sent_string[-1]=='.':
        #    sent_string=sent_string[:-1]
        prediction = candidate.digest_text(sent_string)['neural']
        prediction['stimulus_id'] = 'presentation', np.unique(stim['stimulus_id'].values)
        predictions.append(prediction)
    predictions = xr.concat(predictions, dim='presentation')
    #pred=torch.tensor(predictions.values,device=device)
    #(U,S,V)=torch.pca_lowrank(pred,q=500)
    #var_explained_curve=torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    #plt.plot(var_explained_curve.cpu())
    #plt.show()
    model_save_path=Path(OUTPUT_DIR,candidate.identifier+'_dataset-ud_sentencez_data_token_filter_v3_brainscore_WO_ANNSet1.pkl')
    with open(model_save_path.__str__(), 'wb') as f:
        pickle.dump(predictions, f)

