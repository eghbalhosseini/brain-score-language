import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
from tqdm import tqdm

# load benchmark


if __name__ == '__main__':
    # load all the benchmarks:
    Pereira243=load_benchmark('Pereira2018.243sentences-linear')
    Pereira384=load_benchmark('Pereira2018.384sentences-linear')
    Pereira384_ds_min=load_benchmark('Pereira2018.384sentences.ds.min-linear')
    Pereira384_ds_max = load_benchmark('Pereira2018.384sentences.ds.max-linear')
    Pereira243_ds_min = load_benchmark('Pereira2018.243sentences.ds.min-linear')
    Pereira243_ds_max = load_benchmark('Pereira2018.243sentences.ds.max-linear')

    Pereira=[Pereira243,Pereira384]
    Pereira_ds_min=[Pereira243_ds_min,Pereira384_ds_min]
    Pereira_ds_max=[Pereira243_ds_max,Pereira384_ds_min]
    # for all models compute benchmarks:
    models = ['roberta-base', 'xlm-mlm-en-2048', 'xlnet-large-cased', 'albert-xxlarge-v2', 'bert-base-uncased',
              'gpt2-xl', 'ctrl']
    model_scores_pereira=[]
    model_scores_pereira_ds_min=[]
    model_scores_pereira_ds_max=[]
    for model in tqdm(models):
        candidate=load_model(model)
        model_score=[x(candidate) for x in Pereira]
        model_score_ds_min=[x(candidate) for x in Pereira_ds_min]
        model_score_ds_max = [x(candidate) for x in Pereira_ds_max]
        model_scores_pereira.append(model_score)
        model_scores_pereira_ds_min.append(model_score_ds_min)
        model_scores_pereira_ds_max.append(model_score_ds_max)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/model_scores_pereira.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_scores_pereira, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/model_scores_pereira_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_scores_pereira_ds_min, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/model_scores_pereira_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_scores_pereira_ds_max, f)
# load model
# 1.
    model_Pereira=[np.mean([x.values for x in y]) for y in model_scores_pereira]
    model_Pereira_ds_min = [np.mean([x.values for x in y]) for y in model_scores_pereira_ds_min]
    model_Pereira_ds_max = [np.mean([x.values for x in y]) for y in model_scores_pereira_ds_max]




    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.bar(x - width, np.stack(model_Pereira), width, color=np.divide((55, 76, 128), 256),label='Pereira')
    rects2 = ax.bar(x , np.stack(model_Pereira_ds_min).squeeze(), width, label='Pereira_Ds_min (N=100)',color=np.divide((188, 80, 144), 255))
    rects3 = ax.bar(x + width , np.stack(model_Pereira_ds_max).squeeze(), width, label='Peireira_Ds_max (N=100)',
                    color=np.divide((255, 128, 0), 255))
    # ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
    #             yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
    #             color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title('Comparison of model performance between Pereira with 2 different optimization\n'
                 'in each experiment (243 and 384) 100 samples were selected for ds_ma and ds_min \n'
                 'plots are the average of two experiment under each condition')
    ax.set_xticks(x, models)
    ax.set_ylim((-.1, 1.1))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    #fig.tight_layout()

    fig.show()
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_ds_min_ds_max_comp.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_ds_min_ds_max_comp.eps')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
