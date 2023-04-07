import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == '__main__':
    ANNSet1 = load_benchmark('ANNSet1_fMRI.train.language_top_90-linear')
    models = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-base-uncased','gpt2-xl','ctrl']
    for model in models:
        candidate = load_model(f'{model}')
        benchmark_id=ANNSet1.identifier.replace('.','_')
        save_path=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/{candidate.identifier}_{benchmark_id}_scores.pkl')
        if not save_path.exists():
            model_score = ANNSet1(candidate)
            with open(save_path.__str__(), 'wb') as f:
                pickle.dump(model_score, f)
        else:
            model_score=pd.read_pickle(save_path)
        # reorder scores based on layer numbers
        fig = plt.figure(figsize=(11, 8))
        for id, grp in enumerate(model_score.raw.groupby('subject')):
            ax = None
            ax = plt.subplot(3, 3, id + 1, frameon=True, sharex=ax)
            x = grp[1].tval_lang.values.flatten()
            y = grp[1].mean('split').values.flatten()
            ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k", c='r')
            b, a = np.polyfit(x, y, deg=1)
            xseq = np.linspace(0, 1, num=100)
            ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
            ax.set_title(grp[0])
        plt.tight_layout()
        ax.set_xlabel('tval')
        ax.set_ylabel('variance explained')

        save_loc = Path(
            f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score_vs_tvals.png')
        fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto', backend=None)

        width = 0.7 # the width of the bars
        fig = plt.figure(figsize=(11, 8))
        d = {'tval': model_score.raw.tval_lang,
             'vox_corr': model_score.raw.repetition_corr,
             'vox_score': model_score.raw.mean('split').values.flatten(),
             'subject': model_score.raw.subject}
        df1 = pd.DataFrame(d)

        fig = plt.figure(figsize=(8, 11))
        ax = plt.axes((.1, .45, .55, .45 * (8 / 11)))
        sns.scatterplot(data=df1, x='tval', y='vox_score', hue='subject',s=25)
        p2 = plt.axhline(y=0, color=(.5,.5,.5),zorder=0,linestyle='--')
        xlims=ax.get_xlim()
        ylims=ax.get_ylim()
        regr = linear_model.LinearRegression()
        regr.fit( np.stack([df1.tval.values,
                           np.ones(df1.tval.shape)]).transpose(), np.expand_dims(df1.vox_score.values,axis=1))
        tval_predict=np.arange(np.min(model_score.raw.tval_lang.values),np.max(model_score.raw.tval_lang.values),step=0.1)
        tval_predict=np.stack([tval_predict,np.ones(tval_predict.shape)]).transpose()
        y_pred = regr.predict(tval_predict)
        plt.plot(tval_predict, y_pred, color="k", linewidth=3)
        ax = plt.axes((.1, .8, .55, .15 * (8 / 11)))
        sns.histplot(data=df1, x="tval",color='#3399ff')
        ax.set_xlim(xlims)
        ax.set_xlabel('')
        ax.set_xticklabels('')
        ax.set_title(f'model: {candidate.identifier}\n benchmark: {benchmark_id},\nregression slope:{regr.coef_[0][0]:.5f}')
        ax = plt.axes((.68, .45, .2, .45 * (8 / 11)))
        sns.histplot(data=df1, y="vox_score", color='#3399ff')
        ax.set_ylabel('')
        ax.set_yticklabels('')
        ax.set_ylim(ylims)
        df1_sort=df1.sort_values('tval')
        tval=np.expand_dims(df1_sort.tval.values,axis=1)
        vox_score_sort=np.expand_dims(df1_sort.vox_score.values,axis=1)
        ax = plt.axes((.1, .1, .55, .35 * (8 / 11)))
        ax.plot(np.flipud(vox_score_sort)[:1000],linewidth=.5)
        ax.set_xlabel('Voxel ID, sorted by langloc')
        ax.set_ylabel('Pearson correlation (vox_score)')
        #fig.show()
        save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_aggregate_score_vs_tvals.png')
        fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
        save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_aggregate_score_vs_tvals.eps')
        fig.savefig(save_loc.__str__(),format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
