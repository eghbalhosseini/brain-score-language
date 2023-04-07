import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark, load_model,ArtificialSubject
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
import seaborn as sns
from glob import glob
import re
from tqdm import tqdm
import xarray as xr
import torch
import scipy
from scipy.spatial.distance import pdist,squareform

RESULTCACHING_HOME = '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/'
OUTPUT_DIR='/om2/user/ehoseini/MyData/brain-score-language/output'
ANALYSIS_DIR='/om2/user/ehoseini/MyData/brain-score-language/analysis'
SENT_SAMPLING_DIR=''
# author : chatGPT
def sort_strings_by_number(strings, pattern=r'\d+.pkl'):
    # First, we'll create a list of tuples where each tuple contains the original
    # string and the number that appears in it
    string_tuples = []
    remove_char = lambda s: ''.join(c for c in s if c.isdigit())
    for s in strings:
        # Find the first group of digits in the string
        match = re.search(pattern, s)
        if match:
            # If a group of digits was found, extract it and convert it to an integer

            num = int(remove_char(match.group(0)))
            # Add the tuple to the list
            string_tuples.append((s, num))
        else:
            # If no digits were found, add the tuple with a default value of 0
            string_tuples.append((s, 0))

    # Now we'll sort the list of tuples by the number
    string_tuples.sort(key=lambda x: x[1])

    # Finally, we'll extract just the strings from the sorted list of tuples
    return [t[0] for t in string_tuples]
# author : chatGPT
def get_id_from_strings(s, pattern=r'\d+.pkl'):
    # First, we'll create a list of tuples where each tuple contains the original
    # string and the number that appears in it
    remove_char = lambda s: ''.join(c for c in s if c.isdigit())
    # Find the first group of digits in the string
    match = re.search(pattern, s)
    if match:
            # If a group of digits was found, extract it and convert it to an integer
        num = int(remove_char(match.group(0)))
            # Add the tuple to the list

    else:
            # If no digits were found, add the tuple with a default value of 0
        num = None

    return num


if __name__ == '__main__':
    ANNSet1 = load_benchmark('ANNSet1_fMRI.train.language_top_90-linear')
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    layer_ids = (2, 24, 12, 12, 44, 5, 47)
    name = ANNSet1.data.identifier.replace('.', '-')
    brainscore_activation_file=Path(OUTPUT_DIR,f'{name}_brainscore_activations.pkl')
    if brainscore_activation_file.exists():
        brainscore_language_activations=pd.read_pickle(brainscore_activation_file.__str__())
    else:
        brainscore_language_activations = []
        for model, layer in tqdm(model_layers):
            # extract brainscore:
            candidate = load_model(f'{model}')
            candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                             recording_type=ArtificialSubject.RecordingType.fMRI)

            stimuli = ANNSet1.data['stimulus']
            predictions = []
            for stim_id, stim in tqdm(stimuli.groupby('stimulus_id'), desc='digest individual sentences'):
                True
                prediction = candidate.digest_text(str(stim.values))['neural']
                prediction['stimulus_id'] = 'presentation', stim['stimulus_id'].values
                predictions.append(prediction)
            predictions = xr.concat(predictions, dim='presentation')
            brainscore_language_activations.append(predictions)
        with open(brainscore_activation_file.__str__(),'wb') as f:
            pickle.dump(brainscore_language_activations,f)

    # calculate DS for neural_nlp:
    neural_nlp_activation_file = Path(OUTPUT_DIR, f'{name}_neral_nlp_activations.pkl')
    if neural_nlp_activation_file.exists():
        neural_nlp_activations=pd.read_pickle(neural_nlp_activation_file.__str__())
    else:
        neural_nlp_activations=[]
        for model, layer in tqdm(model_layers):
            # extract brainscore:
            activations_files = glob(Path(RESULTCACHING_HOME,
                                          f'identifier={model},stimuli_identifier={ANNSet1.data.identifier}-*.pkl').__str__())
            activations_files = sort_strings_by_number(activations_files, pattern=r'\d+.pkl')
            assert len(activations_files) == 200
            all_sent_activation = []
            for act_file in activations_files:
                sentence_id=get_id_from_strings(act_file,pattern=r'\d+.pkl')
                sent_act = pd.read_pickle(act_file)['data']
                sent_act=sent_act.assign_coords({'stimulus_id':('presentation',np.repeat(sentence_id,sent_act.shape[0]))})
                all_sent_activation.append(sent_act)
            all_sent_activation = xr.concat(all_sent_activation, dim='presentation')
            layer_index=[x==layer for x in all_sent_activation.layer.values]
            layer_act = all_sent_activation.sel(neuroid=layer_index)
            # get last word
            layer_last_word=xr.concat([x[-1,:] for _, x in layer_act.groupby('stimulus_sentence')],dim='presentation')
            # compare brainscore with neural nlp
            layer_last_word=layer_last_word.sortby('stimulus_id')
            neural_nlp_activations.append(layer_last_word)
        with open(neural_nlp_activation_file.__str__(), 'wb') as f:
            pickle.dump(neural_nlp_activations, f)

    model_correspondance=[]
    for idx, brainscore_act in tqdm(enumerate(brainscore_language_activations)):
        # extract brainscore:
        neural_nlp_act=neural_nlp_activations[idx]
        # compare brainscore with neural nlp
        neural_nlp_act=neural_nlp_act.sortby('stimulus_id')
        brainscore_act=brainscore_act.sortby('stimulus_id')

        df1=pd.DataFrame(neural_nlp_act.values.transpose())
        df2=pd.DataFrame(brainscore_act.values.transpose())
        model_correspondance.append(df1.corrwith(df2).values)


    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    for idx, x in enumerate(model_correspondance):
        ax=plt.subplot(3,3,idx+1)
        ax.hist(x,50,edgecolor='k')
        ax.set_title(model_layers[idx][0])
    ax.set_xlabel('correlation')
    plt.tight_layout()
    text = 'correlation between neural-nlp (period) output and brainscore\n' \
           'for 200 sentences in ANNset1'
    fig.text(0.4, 0.2, text, ha='left', va='center',fontsize=12)
    #fig.show()
    save_loc = Path(ANALYSIS_DIR,f'camprison_between_neural_nlp_period_and_brainscore_act_{name}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    # calculate DS for neural_nlp:
    distance_type='correlation'
    neural_nlp_sent_dist=[pdist(x.values,metric=distance_type) for x in neural_nlp_activations]
    brainscore_sent_dist=[pdist(x.values,metric=distance_type) for x in brainscore_language_activations]

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    for idx, x in enumerate(neural_nlp_sent_dist):
        ax = plt.subplot(3, 3, idx + 1)
        ax.hist(x, 50, edgecolor='k',facecolor='r',label='neural nlp',alpha=.5)
        ax.hist(brainscore_sent_dist[idx], 50, edgecolor='k', facecolor='b',label='brainscore',alpha=.5)
        ax.set_title(model_layers[idx][0])
    ax.set_xlabel('correlation')
    ax.legend()
    plt.tight_layout()
    text = 'camparison of sentence dissimilarity in neural-nlp (Period) and brainscore\n' \
           'for 200 sentences in ANNset1'
    fig.text(0.4, 0.2, text, ha='left', va='center', fontsize=12)
    save_loc = Path(ANALYSIS_DIR, f'comparison_between_neural_nlp_period_and_brainscore_sentence_{distance_type}_dissimilarity_{name}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)


    np.mean(pdist(np.stack(neural_nlp_sent_dist),metric=distance_type))
    np.mean(pdist(np.stack(brainscore_sent_dist),metric=distance_type))
    # results dont replicate here :(

    sent_sampling_data = []
    for l_id,(model, layer) in enumerate(model_layers):

        with open(Path('/nese/mit/group/evlab/u/ehoseini/MyData/sent_sampling/',
                    f'ud_sentences_U01_AnnSET1_ordered_for_RDM_{model}_layer_{layer_ids[l_id]}_activation_ave_False.pkl').__str__(),
                  'rb') as f:
            mod_data = pickle.load(f)
            sent_sampling_data.append(mod_data)
    sentences_from_sampling=[]
    for sample_ in sent_sampling_data:
        sentences=[x[1] for x in sample_]
        sentences_from_sampling.append(sentences)

    sentences_from_sampling.append(stimuli.values)
    sent_dat=pd.DataFrame(zip(*sentences_from_sampling),columns=([x[0] for x in model_layers].append('brainscore')))
    sent_dat.to_csv(Path(ANALYSIS_DIR,'sentence_from_sent_sampling.csv'))

    sent_sampling_activation=[np.stack([y[0] for y in x]) for x in sent_sampling_data]


    sent_sampling_sent_dist = [pdist(x, metric=distance_type) for x in sent_sampling_activation]
    np.mean(pdist(np.stack(sent_sampling_sent_dist), metric=distance_type))
    # this yields 1.078
    # in this case try the brainscore with the new set of sentences :
    brainscore_sent_sampling_activations = []
    for model, layer in tqdm(model_layers):
        # extract brainscore:
        candidate = load_model(f'{model}')
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)

        stimuli = sent_dat[0]
        predictions = []
        for stim_id, stim in tqdm(enumerate(stimuli), desc='digest individual sentences'):
            prediction = candidate.digest_text(stim)['neural']
            prediction=prediction.assign_coords({'stimulus_id':('presentation', [stim_id+1])})
            predictions.append(prediction)
        predictions = xr.concat(predictions, dim='presentation')
        brainscore_sent_sampling_activations.append(predictions)

    model_correspondance=[]
    for idx,sent_act in enumerate(sent_sampling_activation):
        df1=pd.DataFrame(sent_act.transpose())
        df2=pd.DataFrame(brainscore_sent_sampling_activations[idx].values.transpose())
        df1.corrwith(df2).values
        model_correspondance.append(df1.corrwith(df2).values)
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    for idx, x in enumerate(model_correspondance):
        ax = plt.subplot(3, 3, idx + 1)
        ax.hist(x, 50, edgecolor='k')
        ax.set_title(model_layers[idx][0])
    ax.set_xlabel('correlation')
    plt.tight_layout()
    fig.show()


    #brainscore_sent_sampling_sent_dist = [pdist(x, metric=distance_type) for x in brainscore_sent_sampling_activations]
    #np.mean(pdist(np.stack(brainscore_sent_sampling_sent_dist), metric=distance_type))
    #squareform(pdist(np.stack(brainscore_sent_sampling_sent_dist), metric=distance_type))


    # do a version for word form
    neural_nlp_wordForm_activation_file = Path(OUTPUT_DIR, f'{name}_wordForm_neral_nlp_activations.pkl')
    if neural_nlp_wordForm_activation_file.exists():
        neural_nlp_wordForm_activations=pd.read_pickle(neural_nlp_wordForm_activation_file.__str__())
    else:
        neural_nlp_wordForm_activations=[]
        for model, layer in tqdm(model_layers):
            # extract brainscore:
            True
            activations_files = glob(Path(RESULTCACHING_HOME,
                                          f'identifier={model},stimuli_identifier={ANNSet1.data.identifier}_wordForm-*.pkl').__str__())
            activations_files = sort_strings_by_number(activations_files, pattern=r'\d+.pkl')
            assert len(activations_files) == 200
            all_sent_activation = []
            for act_file in activations_files:
                sentence_id=get_id_from_strings(act_file,pattern=r'\d+.pkl')
                sent_act = pd.read_pickle(act_file)['data']
                sent_act=sent_act.assign_coords({'stimulus_id':('presentation',np.repeat(sentence_id,sent_act.shape[0]))})
                all_sent_activation.append(sent_act)
            all_sent_activation = xr.concat(all_sent_activation, dim='presentation')
            layer_index=[x==layer for x in all_sent_activation.layer.values]
            layer_act = all_sent_activation.sel(neuroid=layer_index)
            # get last word
            layer_last_word=xr.concat([x[-1,:] for _, x in layer_act.groupby('stimulus_sentence')],dim='presentation')
            # compare brainscore with neural nlp
            layer_last_word=layer_last_word.sortby('stimulus_id')
            neural_nlp_wordForm_activations.append(layer_last_word)
        with open(neural_nlp_wordForm_activation_file.__str__(), 'wb') as f:
            pickle.dump(neural_nlp_wordForm_activations, f)

    #
    neural_nlp_wordForm_sent_sampling_sent_dist = [pdist(x, metric=distance_type) for x in neural_nlp_wordForm_activations]
    np.mean(pdist(np.stack(neural_nlp_wordForm_sent_sampling_sent_dist), metric=distance_type))

    model_correspondance = []
    for idx, sent_act in tqdm(enumerate(neural_nlp_wordForm_activations)):
        df1 = pd.DataFrame(sent_act.values.transpose())
        df2 = pd.DataFrame(neural_nlp_activations[idx].values.transpose())
        df1.corrwith(df2).values
        model_correspondance.append(df1.corrwith(df2).values)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    for idx, x in enumerate(model_correspondance):
        ax = plt.subplot(3, 3, idx + 1)
        ax.hist(x, 50, edgecolor='k')
        ax.set_title(model_layers[idx][0])
    ax.set_xlabel('correlation')
    plt.tight_layout()

    text = 'correlation between neural-nlp (Peiord) and \n'\
           'neural-nlp (wordForm) for ANNset1 sentences'
    fig.text(0.4, 0.2, text, ha='left', va='center', fontsize=12)
    fig.show()
    save_loc = Path(ANALYSIS_DIR, f'camprison_between_neural_nlp_period_and_neural_nlp_wordFrom_{name}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    distance_type = 'correlation'
    neural_nlp_sent_dist = [pdist(x.values, metric=distance_type) for x in neural_nlp_activations]
    neural_nlp_wordForm_sent_dist = [pdist(x.values, metric=distance_type) for x in neural_nlp_wordForm_activations]

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    for idx, x in enumerate(neural_nlp_sent_dist):
        ax = plt.subplot(3, 3, idx + 1)
        ax.hist(x, 50, edgecolor='k', facecolor='r', label='neural nlp', alpha=.5)
        ax.hist(neural_nlp_wordForm_sent_dist[idx], 50, edgecolor='k', facecolor='b', label='neural nlp (wordForm)', alpha=.5)
        ax.set_title(model_layers[idx][0])
    ax.set_xlabel('correlation')
    ax.legend()
    plt.tight_layout()
    fig.show()
    text = 'camparison of sentence dissimilarity in neural-nlp (period) and \n' \
           ' neural-nlp (wordForm) for 200 sentences in ANNset1'
    fig.text(0.4, 0.2, text, ha='left', va='center', fontsize=12)
    fig.show()
    save_loc = Path(ANALYSIS_DIR,
                    f'comparison_between_neural_nlp_and_neural_nlp_wordForm_sentence_{distance_type}_dissimilarity_{name}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)
