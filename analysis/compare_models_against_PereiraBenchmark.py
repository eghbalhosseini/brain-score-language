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
    kk=0
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


    # plot sampler resutls

    Pereria384_ds_max = load_benchmark(f'Pereira2018.384sentences.ds.max.150-linear')
    Pereria384_ds_min = load_benchmark(f'Pereira2018.384sentences.ds.min.150-linear')
    Pereria384_ds_max_rand =load_benchmark(f'Pereira2018.384sentences.ds.max.150.rand.0-linear')


    Pereria243_ds_max = load_benchmark(f'Pereira2018.243sentences.ds.max.150-linear')
    Pereria243_ds_max_rand =load_benchmark(f'Pereira2018.243sentences.ds.max.150.rand.0-linear')
    Pereria243_ds_min = load_benchmark(f'Pereira2018.243sentences.ds.min.150-linear')
    Pereria243_ds_min_rand = [load_benchmark(f'Pereira2018.243sentences.ds.min.{x}.rand.0-linear') for x in [150, 200]]

    xx=pd.read_pickle(Pereria384_ds_max.sampler)
    xx_min = pd.read_pickle(Pereria384_ds_min.sampler)
    xx_rand = pd.read_pickle(Pereria384_ds_max_rand.sampler)

    yy = pd.read_pickle(Pereria243_ds_max.sampler)
    yy_min = pd.read_pickle(Pereria243_ds_min.sampler)
    yy_rand=pd.read_pickle(Pereria243_ds_max_rand.sampler)



    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.plot(np.cumsum(xx["Ds_trajectory"][:,1]),(xx["Ds_trajectory"][:,2]), color='r')
    rects1 = ax.plot(np.cumsum(xx_min["Ds_trajectory"][:, 1]), 2-(xx_min["Ds_trajectory"][:, 2]), color='b')

    rects1 = ax.plot(np.cumsum(yy["Ds_trajectory"][:, 1]), (yy["Ds_trajectory"][:, 2]), color='r')
    rects1 = ax.plot(np.cumsum(yy_min["Ds_trajectory"][:, 1]), 2 - (yy_min["Ds_trajectory"][:, 2]), color='b')

    #ax.set_xscale('log')
    fig.show()

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.2, .1, .08, .45))
    ax.scatter(.1 * np.random.normal(size=(np.asarray(xx_rand['Ds_trajectory']).shape)) + 0, np.asarray(xx_rand['Ds_trajectory']),
               color=(.6, .6, .6), s=2, alpha=.3)
    ax.scatter(0, np.asarray(xx_rand['Ds_trajectory']).mean(), color=np.divide((55, 76, 128), 256), s=50, label='random',edgecolor='k')
    ax.scatter(0, 2-xx_min['Ds'], color=np.divide((188, 80, 144), 255), s=50, label='Ds_min',edgecolor='k')
    ax.scatter(0, xx['Ds'], color=np.divide((255, 128, 0), 255), s=50, label='Ds_max',edgecolor='k')

    ax.scatter(.1 * np.random.normal(size=(np.asarray(yy_rand['Ds_trajectory']).shape)) + 2,
               np.asarray(yy_rand['Ds_trajectory']),
               color=(.6, .6, .6), s=2, alpha=.3)
    ax.scatter(2, np.asarray(yy_rand['Ds_trajectory']).mean(), color=np.divide((55, 76, 128), 256), s=50,edgecolor='k')
    ax.scatter(2, 2 - yy_min['Ds'], color=np.divide((188, 80, 144), 255), s=50,edgecolor='k')
    ax.scatter(2, yy['Ds'], color=np.divide((255, 128, 0), 255), s=50,edgecolor='k')

    tick_l = ['384','243']
    tick = [0,2]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_xlim((-.8, 2.8))
    ax.set_ylim((.5, 1.1))
    ax.set_xticks(tick)
    ax.set_xticklabels(tick_l)
    ax.set_xlabel('# Sentences')
    ax.legend(bbox_to_anchor=(1.7, 1.1), frameon=True)
    ax.set_ylabel(r'$D_s$')

    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    ax = plt.axes((.35, .1, .3, .15))
    rects1 = ax.plot(np.cumsum(xx["Ds_trajectory"][:,1]),(xx["Ds_trajectory"][:,2]), color=np.divide((255, 128, 0), 255),linewidth=3)
    rects1 = ax.plot(np.cumsum(xx_min["Ds_trajectory"][:, 1]), 2-(xx_min["Ds_trajectory"][:, 2]), color=np.divide((188, 80, 144), 255),linewidth=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_ylim((.5, 1.1))
    ax.set_title('384 sentence optimization (n=150)')

    ax = plt.axes((.35, .35, .3, .15))
    rects1 = ax.plot(np.cumsum(yy["Ds_trajectory"][:,1]),(yy["Ds_trajectory"][:,2]), color=np.divide((255, 128, 0), 255),linewidth=3)
    rects1 = ax.plot(np.cumsum(yy_min["Ds_trajectory"][:, 1]), 2-(yy_min["Ds_trajectory"][:, 2]), color=np.divide((188, 80, 144), 255),linewidth=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_ylim((.5, 1.1))
    ax.set_title('243 sentence optimization (n=150)')

    fig.show()

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_optim_res.png')
    fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/Pereira_243_ds_min_ds_max_optim_res.eps')
    fig.savefig(save_loc.__str__(), format='eps', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto',
                edgecolor='auto', backend=None)


