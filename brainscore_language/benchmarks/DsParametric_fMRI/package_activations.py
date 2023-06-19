from pathlib import Path
import os
from glob import glob
import pandas as pd
import xarray as xr
import pickle
import numpy as np
from brainscore_language import load_dataset, load_metric
ACTIVATON_DIR= '/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/activations/'
if __name__ == '__main__':
        model_id_list=['xlnet-large-cased',
                        'xlm-mlm-en-2048'
                     ,'albert-xxlarge-v2'
         ,'bert-large-uncased-whole-word-masking'
         ,'roberta-base'
         ,'gpt2-xl'
         ,'ctrl']
        exprs=['DsParametricfMRI_max','DsParametricfMRI_min','DsParametricfMRI_random']
        data=load_dataset('DsParametric_fMRI.language_top_90')
        for expr in exprs:
            for model_id in model_id_list:
                save_path = Path(ACTIVATON_DIR) / f'model={model_id}_stimuli={expr}.pkl'
                if save_path.exists():
                    continue
                else:
                    pattern = os.path.join(
                     '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored',
                         f'identifier={model_id},stimuli_identifier={expr}*.pkl')
                    sent_set = glob(pattern)
                    assert(len(sent_set)==80)
                    # make a list and load all the data
                    a_list=[]
                    stim_group=expr.split('_')[-1]
                    data_condition=data[data['stim_group']==stim_group]
                    for kk in sent_set:
                        a=pd.read_pickle(kk)
                        a=a['data']
                        # reorder the data based on layer
                        #a_layer=[x for idx, x in a.groupby('layer')]
                        sent=np.unique(a.stimulus_sentence)[0]
                        # find sent data.sentence.values using np.where
                        sent_loc=np.argwhere(data_condition.sentence.values==sent)
                        assert(len(sent_loc)==1)
                        # find stim_name for the sentence
                        stim_name=data_condition.stim_name.values[sent_loc[0][0]]
                        # find stim_group for the sentence
                        stim_group=data_condition.stim_group.values[sent_loc[0][0]]
                        # find stimulus_id for the sentence
                        stimulus_id=data_condition.stimulus_id.values[sent_loc[0][0]]
                        #
                        # assign a new coordiante to a name stim_name in dimension presentation
                        a=a.assign_coords({'stim_name': ('presentation', np.repeat(stim_name,a.shape[0])),
                                           'stim_group': ('presentation', np.repeat(stim_group,a.shape[0])),
                                           'stimulus_id': ('presentation', np.repeat(stimulus_id,a.shape[0]))})
                        a_list.append(a)
                    a_group=xr.concat(a_list,dim='presentation')
                    # save the a_group in activation DIR

                    with open(save_path, 'wb') as f:
                        pickle.dump(a_group, f)




