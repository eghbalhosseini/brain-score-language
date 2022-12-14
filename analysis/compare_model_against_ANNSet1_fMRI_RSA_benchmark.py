import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re

if __name__ == '__main__':
    ANNSet1_RSA=_ANNSet1_fMRI_ExperimentRSA(atlas='train.language_top_90',ceiling_s3_kwargs=None)

    models = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-base-uncased','gpt2-xl','ctrl']
    for model in models:
        candidate = load_model(f'{model}')
        benchmark_id=ANNSet1_RSA.identifier.replace('.','_')
        model_score = ANNSet1_RSA(candidate)

        #fig.show()
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score.eps')
        fig.savefig(save_loc.__str__(),format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
