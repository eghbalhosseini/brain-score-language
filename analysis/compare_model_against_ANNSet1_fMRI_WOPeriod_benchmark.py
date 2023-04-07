import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re

if __name__ == '__main__':
    ANNSet1 = load_benchmark('ANNSet1_fMRI_WOPeriod.train.language_top_90-linear')
    models = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-base-uncased','gpt2-xl','ctrl']
    model_scores=[]
    for model in models:
        True
        candidate = load_model(f'{model}')
        # benchmark_id=ANNSet1.identifier.replace('.','_')

        model_score = ANNSet1(candidate)

        model_scores.append(model_score)



    model_ANNSet1 = [x.values[0] for x in model_scores]
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
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title('model scores with no period')
    ax.set_xticks(x, labels)
    ax.set_ylim((-.1, 1.1))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    # fig.tight_layout()

    fig.show()
    save_loc = Path(
        f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/ANNSet1_WOperiod_performance_all_models.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(
        f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/ANNSet1_WOperiod_performance_all_models.eps')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

        #
        # save_path=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/{candidate.identifier}_{benchmark_id}_scores.pkl')
        # if not save_path.exists():
        #     model_score = ANNSet1(candidate)
        #     with open(save_path.__str__(), 'wb') as f:
        #         pickle.dump(model_score, f)
        # else:
        #     model_score=pd.read_pickle(save_path)
        # # reorder scores based on layer numbers
        # if model=='albert-xxlarge-v2':
        #     layer_loc=[int(re.findall(r'groups.\d+',x)[0].lstrip('groups.'))+1 if len(re.findall(r'groups.\d+',x))>0 else 0 for x in model_score.layer.values ]
        #     reorder=np.argsort(layer_loc)
        # else:
        #     ordered_layers=[x[0] for x in candidate.basemodel.named_modules()]
        #     layer_loc=[ordered_layers.index(x) for x in model_score.layer.values]
        #     reorder=np.argsort(layer_loc)
        # width = 0.7 # the width of the bars
        # fig = plt.figure(figsize=(11, 8))
        # ax = plt.axes((.1, .4, .55, .35))
        # x=np.arange(model_score.shape[1])
        # y=(np.stack(model_score[0,:]).squeeze())[reorder]
        # y_err=(np.stack(model_score[1,:]).squeeze())[reorder]
        # layer_name=(model_score.layer.values)[reorder]
        # rects1 = ax.bar(x , y, width,color=np.divide((188, 80, 144), 255))
        # ax.errorbar(x , y,yerr=y_err, linestyle='',color='k')
        # ax.axhline(y=0, color='k', linestyle='-')
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Pearson correlation')
        # ax.set_title(f'{candidate.identifier} performance on {benchmark_id}')
        # ax.set_xticks(x)
        # ax.set_ylim((-.1, 1.1))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # ax.set_xticklabels(layer_name, rotation=90, fontsize=8)
        # ax.set_ylabel('Pearson corr')
        # #fig.show()
        # save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score.png')
        # fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
        #
        # save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score.eps')
        # fig.savefig(save_loc.__str__(),format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
