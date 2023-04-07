import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model, ArtificialSubject, load_dataset
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
UD_PARENT='/om/weka/evlab/ehoseini//MyData/Universal Dependencies 2.6/'
SAVE_DIR='/om2/user/ehoseini/MyData/brain-score-language/dataset'
OUTPUT_DIR='/om2/user/ehoseini/MyData/brain-score-language/output'

parser = argparse.ArgumentParser(description='packaging first level files for subject')
parser.add_argument('model_id', type=str)
args=parser.parse_args()

if __name__ == '__main__':
    model_id = int(args.model_id)

    models = ['roberta-base', 'xlm-mlm-en-2048', 'xlnet-large-cased', 'albert-xxlarge-v2', 'bert-large-uncased-whole-word-masking',
              'gpt2-xl', 'ctrl']
    model=models[model_id]
    candidate = load_model(f'{model}')
    candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                     recording_type=ArtificialSubject.RecordingType.fMRI)
    # pereira 243
    Pereira243_benchmark = load_benchmark('Pereira2018.243sentences-linear')
    stimuli = Pereira243_benchmark.data['stimulus']
    passages = Pereira243_benchmark.data['passage_label'].values
    predictions = []
    for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
        passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
        passage_stimuli = stimuli[passage_indexer]
        passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
        passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
        predictions.append(passage_predictions)
    predictions = xr.concat(predictions, dim='presentation')
    data_id=Pereira243_benchmark.data.identifier.replace('.','_')


    model_save_path=Path(OUTPUT_DIR,candidate.identifier+f'_dataset-{data_id}.pkl')

    with open(model_save_path.__str__(), 'wb') as f:
        pickle.dump(predictions, f)

    # save the data
    save_path = Path(SAVE_DIR, f'{data_id}.pkl')
    if not save_path.exists():
        perei=Pereira243_benchmark.data
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(perei, f)

    # pereira 384
    Pereira384_benchmark = load_benchmark('Pereira2018.384sentences-linear')
    stimuli = Pereira384_benchmark.data['stimulus']
    passages = Pereira384_benchmark.data['passage_label'].values
    predictions = []
    for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
        passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
        passage_stimuli = stimuli[passage_indexer]
        passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
        passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
        predictions.append(passage_predictions)
    predictions = xr.concat(predictions, dim='presentation')
    data_id = Pereira384_benchmark.data.identifier.replace('.', '_')
    model_save_path = Path(OUTPUT_DIR, candidate.identifier + f'_dataset-{data_id}.pkl')

    with open(model_save_path.__str__(), 'wb') as f:
        pickle.dump(predictions, f)

    save_path = Path(SAVE_DIR, f'{data_id}.pkl')
    if not save_path.exists():
        perei = Pereira384_benchmark.data
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(perei, f)
