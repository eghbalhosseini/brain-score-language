import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
from tqdm import tqdm

# load benchmark


if __name__ == '__main__':
    # load all the benchmarks:
    #Pereira243=load_benchmark('Pereira2018.243sentences-linear')
    #Pereira384=load_benchmark('Pereira2018.384sentences-linear')


    # for all models compute benchmarks:
    models = ['roberta-base', 'xlm-mlm-en-2048', 'xlnet-large-cased', 'albert-xxlarge-v2', 'bert-base-uncased',
              'gpt2-xl', 'ctrl']
    model_scores_pereira_384_ds_max=[]
    model_scores_pereira_384_ds_max_rand=[]
    model_scores_pereira_384_ds_min=[]
    model_scores_pereira_384_ds_min_rand=[]

    model_scores_pereira_243_ds_max=[]
    model_scores_pereira_243_ds_max_rand=[]
    model_scores_pereira_243_ds_min=[]
    model_scores_pereira_243_ds_min_rand=[]
    kk_val=[150,200]
    kk=1
    for model in models:
        model_scores_pereira_384_ds_max.append(pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_max.pkl')[kk])
        model_scores_pereira_384_ds_max_rand.append(pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_max_rand.pkl')[kk])
        model_scores_pereira_384_ds_min.append(pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_min.pkl')[kk])
        model_scores_pereira_384_ds_min_rand.append(
            pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_min_rand.pkl')[kk])

        model_scores_pereira_243_ds_max.append(
            pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_max.pkl')[kk])
        model_scores_pereira_243_ds_max_rand.append(
            pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_max_rand.pkl')[kk])
        model_scores_pereira_243_ds_min.append(
            pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_min.pkl')[kk])
        model_scores_pereira_243_ds_min_rand.append(
            pd.read_pickle(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_min_rand.pkl')[kk])

    # load model
# 1.
    model_Pereira=np.stack([model_scores_pereira_243_ds_max_rand,model_scores_pereira_384_ds_max]).squeeze().mean(axis=0)
    model_Pereira_ds_min = np.stack([model_scores_pereira_243_ds_min, model_scores_pereira_384_ds_min]).squeeze().mean(axis=0)
    model_Pereira_ds_max =np.stack([model_scores_pereira_243_ds_max, model_scores_pereira_384_ds_max]).squeeze().mean(axis=0)
    #model_Pereira_rand=[np.mean([x.values for x in y]) for y in model_scores_pereira_384_ds_max]
    #model_Pereira_ds_min = [np.mean([x.values for x in y]) for y in model_scores_pereira_ds_min]
    #model_Pereira_ds_max = [np.mean([x.values for x in y]) for y in model_scores_pereira_ds_max]




    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.bar(x , np.stack(model_Pereira), width, color=np.divide((55, 76, 128), 256),label='Pereira')
    rects2 = ax.bar(x - width , np.stack(model_Pereira_ds_min).squeeze(), width, label='Pereira_Ds_min (N=100)',color=np.divide((188, 80, 144), 255))
    rects3 = ax.bar(x + width , np.stack(model_Pereira_ds_max).squeeze(), width, label='Peireira_Ds_max (N=100)',
                    color=np.divide((255, 128, 0), 255))
    # ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
    #             yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
    #             color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
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
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_ds_min_ds_max_comp_{kk_val[kk]}.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_ds_min_ds_max_comp{kk_val[kk]}.eps')
    fig.savefig(save_loc.__str__(), format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


    # do it per exeperiment
    score_p243_rand_mean=[np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in model_scores_pereira_243_ds_max_rand]
    score_p243_rand_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_243_ds_max_rand]

    score_p243_min_mean = [np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_243_ds_min]
    score_p243_min_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                           model_scores_pereira_243_ds_min]

    score_p243_max_mean = [np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_243_ds_max]
    score_p243_max_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                           model_scores_pereira_243_ds_max]


    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.bar(x, np.stack(score_p243_rand_mean), width, color=np.divide((55, 76, 128), 256), label=f'Pereira (N={kk_val[kk]})')
    ax.errorbar(x, score_p243_rand_mean,yerr=score_p243_rand_std, linestyle='',color='k')

    rects2 = ax.bar(x - width, score_p243_min_mean, width, label=f'Pereira_Ds_min (N={kk_val[kk]})',
                    color=np.divide((188, 80, 144), 255))
    ax.errorbar(x - width, score_p243_min_mean, yerr=score_p243_min_std, linestyle='', color='k')
    rects3 = ax.bar(x + width, score_p243_max_mean, width, label=f'Peireira_Ds_max (N={kk_val[kk]})',
                    color=np.divide((255, 128, 0), 255))
    ax.errorbar(x + width, score_p243_max_mean, yerr=score_p243_min_std, linestyle='', color='k')
    # ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
    #             yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
    #             color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')

    ax.set_xticks(x, models)
    ax.set_ylim((-.1, .4))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    # fig.tight_layout()

    fig.show()

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_comp_{kk_val[kk]}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_comp_{kk_val[kk]}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)



# do it per exeperiment
    score_p384_rand_mean=[np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in model_scores_pereira_384_ds_max_rand]
    score_p384_rand_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_384_ds_max_rand]

    score_p384_min_mean = [np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_384_ds_min]
    score_p384_min_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                           model_scores_pereira_384_ds_min]

    score_p384_max_mean = [np.mean(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                            model_scores_pereira_384_ds_max]
    score_p384_max_std = [np.std(x.raw.raw.mean('split').groupby('subject').median().values) for x in
                           model_scores_pereira_384_ds_max]


    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.bar(x, np.stack(score_p384_rand_mean), width, color=np.divide((55, 76, 128), 256), label=f'Pereira (N={kk_val[kk]})')
    ax.errorbar(x, score_p384_rand_mean,yerr=score_p384_rand_std, linestyle='',color='k')

    rects2 = ax.bar(x - width, score_p384_min_mean, width, label=f'Pereira_Ds_min (N={kk_val[kk]})',
                    color=np.divide((188, 80, 144), 255))
    ax.errorbar(x - width, score_p384_min_mean, yerr=score_p384_min_std, linestyle='', color='k')
    rects3 = ax.bar(x + width, score_p384_max_mean, width, label=f'Peireira_Ds_max (N={kk_val[kk]})',
                    color=np.divide((255, 128, 0), 255))
    ax.errorbar(x + width, score_p384_max_mean, yerr=score_p384_min_std, linestyle='', color='k')
    # ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
    #             yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
    #             color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')

    ax.set_xticks(x, models)
    ax.set_ylim((-.1, .4))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    # fig.tight_layout()

    fig.show()

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_comp_{kk_val[kk]}.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_comp_{kk_val[kk]}.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)

