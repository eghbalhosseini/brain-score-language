import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re

if __name__ == '__main__':
    #ANNSet1_RSA=_ANNSet1_fMRI_ExperimentRSA(atlas='train.language_top_90',ceiling_s3_kwargs=None)
    ANNSet1_RSA = load_benchmark('ANNSet1_fMRI.train.language_top_90-rsa')
    benchmark_id = ANNSet1_RSA.identifier.replace('.', '_')
    models = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-large-uncased','gpt2-xl','ctrl']
    model_scores=[]
    for model in models:
        candidate = load_model(f'{model}')

        model_score = ANNSet1_RSA(candidate)
        model_scores.append(model_score)
        #fig.show()

    model_ANNSet1 = [x.values[0] for x in model_scores]
    model_ANNSet1_noise=[x.attrs['noise_ceiling'] for x in model_scores][0]
    model_ANNSet1_err = [x.values[1] for x in model_scores]
    labels = models
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects2 = ax.bar(x + width / 2, np.stack(model_ANNSet1).squeeze(), width, label='ANNSet1_fMRI',
                    color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
                yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
                color='k')

    ax.axhline(y=0, color='k', linestyle='-')
    ax.axhline(y=np.min(model_ANNSet1_noise), color='k', linestyle='--')
    ax.axhline(y=np.max(model_ANNSet1_noise), color='k', linestyle='--')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation')
    ax.set_title('Model RSA comparisons')
    ax.set_xticks(x, labels)
    ax.set_ylim((-.1, np.max(model_ANNSet1_noise)+.1))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    fig.show()
    save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_models_{benchmark_id}_score.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_models_{benchmark_id}_score.eps')
    fig.savefig(save_loc.__str__(),format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
