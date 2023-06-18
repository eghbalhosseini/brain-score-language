from pathlib import Path
import os
from glob import glob
import pandas as pd
import xarray as xr
import pickle
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
        for expr in exprs:
            for model_id in model_id_list:
                pattern = os.path.join(
                 '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored',
                     f'identifier={model_id},stimuli_identifier={expr}*.pkl')
                sent_set = glob(pattern)
                assert(len(sent_set)==80)
                # make a list and load all the data
                a_list=[]
                for kk in sent_set:
                    a=pd.read_pickle(kk)
                    a=a['data']
                    # reorder the data based on layer
                    #a_layer=[x for idx, x in a.groupby('layer')]

                    a_list.append(a)
                a_group=xr.concat(a_list,dim='presentation')
                # save the a_group in activation DIR
                save_path=Path(ACTIVATON_DIR)/f'model={model_id}_stimuli={expr}.pkl'
                with open(save_path, 'wb') as f:
                    pickle.dump(a_group, f)




