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


if __name__ == '__main__':

    Pereira243_benchmark = load_benchmark('Pereira2018.243sentences-linear')
    Pereira243_ds_max_benchmark = load_benchmark('Pereira2018.243sentences.ds.max-linear')
    Pereira243_ds_min_benchmark = load_benchmark('Pereira2018.243sentences.ds.min-linear')
    Pereira384_ds_max_benchmark = load_benchmark('Pereira2018.384sentences.ds.max-linear')
    Pereira384_ds_min_benchmark = load_benchmark('Pereira2018.384sentences.ds.min-linear')
    print('roberta-base')
    candidate = load_model('roberta-base')
    # load previous Pereia score
    roberta_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p234.pkl')
    roberta_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p384.pkl')
    # compute ds_min and ds_max scores
    roberta_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    roberta_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    roberta_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    roberta_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(roberta_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(roberta_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(roberta_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/roberta_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(roberta_score_p384_ds_min, f)
    print('xlm-mlm-en-2048')
    candidate = load_model('xlm-mlm-en-2048')
    xlm_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p234.pkl')
    xlm_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p384.pkl')
    xlm_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    xlm_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    xlm_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    xlm_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlm_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlm_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlm_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlm_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlm_score_p384_ds_min, f)

    print('xlnet-large-cased')
    candidate = load_model('xlnet-large-cased')
    xlnet_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p234.pkl')
    xlnet_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p384.pkl')
    xlnet_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    xlnet_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    xlnet_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    xlnet_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlnet_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlnet_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlnet_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/xlnet_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(xlnet_score_p384_ds_min, f)

    print('albert-xxlarge-v2')
    candidate = load_model('albert-xxlarge-v2')
    albert_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p234.pkl')
    albert_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p384.pkl')
    albert_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    albert_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    albert_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    albert_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(albert_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(albert_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(albert_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/albert_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(albert_score_p384_ds_min, f)

    print('bert-base-uncased')
    candidate = load_model('bert-base-uncased')
    bert_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p234.pkl')
    bert_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p384.pkl')
    bert_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    bert_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    bert_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    bert_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(bert_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(bert_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(bert_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/bert_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(bert_score_p384_ds_min, f)

    print('gpt2-xl')
    candidate = load_model('gpt2-xl')
    gpt2_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p234.pkl')
    gpt2_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p384.pkl')
    gpt2_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    gpt2_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    gpt2_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    gpt2_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(gpt2_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(gpt2_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(gpt2_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/gpt2_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(gpt2_score_p384_ds_min, f)

    print('ctrl')
    candidate = load_model('ctrl')
    ctrl_score_p234 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p234.pkl')
    ctrl_score_p384 = pd.read_pickle(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p384.pkl')
    ctrl_score_p234_ds_max = Pereira243_ds_max_benchmark(candidate)
    ctrl_score_p384_ds_max = Pereira384_ds_max_benchmark(candidate)
    ctrl_score_p234_ds_min = Pereira243_ds_min_benchmark(candidate)
    ctrl_score_p384_ds_min = Pereira384_ds_min_benchmark(candidate)

    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p234_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(ctrl_score_p234_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(ctrl_score_p384_ds_max, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p234_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(ctrl_score_p234_ds_min, f)
    save_dir = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ctrl_score_p384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(ctrl_score_p384_ds_min, f)

    # sampler=f'{OUTPUT_DIR}/{sampler}'
    # bench=Pereira2018_384sentences_ds_max()
    # model='roberta-base'
    # candidate = load_model(f'{model}')
    # candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
    #                                  recording_type=ArtificialSubject.RecordingType.fMRI)
    #
    # stimuli = bench.data['stimulus']
    # passages = bench.data['passage_label'].values
    # predictions = []
    # for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
    #     passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
    #     passage_stimuli = stimuli[passage_indexer]
    #     passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
    #     passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
    #     predictions.append(passage_predictions)
    # predictions = xr.concat(predictions, dim='presentation')
