import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model, ArtificialSubject
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import xarray as xr
import argparse
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
UD_PARENT='/om/user/ehoseini/MyData/Universal Dependencies 2.6/'
SAVE_DIR='/om2/user/ehoseini/MyData/brain-score-language/dataset'
OUTPUT_DIR='/om2/user/ehoseini/MyData/brain-score-language/output'

parser = argparse.ArgumentParser(description='packaging first level files for subject')
parser.add_argument('model_id', type=str)
args=parser.parse_args()

if __name__ == '__main__':
    model_id = int(args.model_id)
    save_path = Path(SAVE_DIR, 'ud_sentencez_data_token_filter_v3_brainscore.pkl')
    if save_path.exists():
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
        ud_sentences_xr = []
        for sent_id,sent_ in tqdm(enumerate(ud_sentence_clean)):
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
        prediction = candidate.digest_text(sent_string)['neural']
        prediction['stimulus_id'] = 'presentation', np.unique(stim['stimulus_id'].values)
        predictions.append(prediction)
    predictions = xr.concat(predictions, dim='presentation')
    model_save_path=Path(OUTPUT_DIR,candidate.identifier+'_dataset-ud_sentencez_data_token_filter_v3_brainscore.pkl')
    with open(model_save_path.__str__(), 'wb') as f:
        pickle.dump(predictions, f)
