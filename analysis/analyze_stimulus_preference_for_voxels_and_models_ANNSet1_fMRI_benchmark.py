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
from scipy.stats import kendalltau
from tqdm import tqdm
import torch
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
#
import sklearn
from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import hamming
# import pysdtw
# import torch
# from tqdm import tqdm
# device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from torchmetrics.functional import pairwise_cosine_similarity,pairwise_linear_similarity,pairwise_euclidean_distance
#
from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#
#
import time
if __name__ == '__main__':
    ANNSet1 = load_benchmark('ANNSet1_fMRI.best.language_top_90_V2-linear')
    models = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-base-uncased','gpt2-xl','ctrl']
    voxs_dat=[]
    for vox_id,vox_dat in tqdm(ANNSet1.data.groupby('neuroid_id')):
        True
        a=np.argsort(vox_dat.values,axis=0).flatten()
        vox_dat=vox_dat.transpose()[:,a]
        voxs_dat.append(vox_dat)

    vox_dtw_dist=np.nan*np.ones((len(voxs_dat),len(voxs_dat)))
    for idx, vox1 in tqdm(enumerate(voxs_dat)):
        for idy, vox2 in enumerate(voxs_dat):
            s1=vox1.stimulus_id.values.astype(np.double)
            s2=vox2.stimulus_id.values.astype(np.double)
            #start = time.process_time()
            #distance = dtw.distance(s1,s2)
            #print(time.process_time() - start)
            d = dtw.distance_fast(s1, s2, use_pruning=True)
            #print(time.process_time() - start)
            vox_dtw_dist[idx,idy]=d

    benchmark_id=ANNSet1.data.identifier.replace('.','-')
    save_path=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/vox_preference_dtw_fast_{benchmark_id}_scores.pkl')
    if not save_path.exists():
        with open(save_path.__str__(), 'wb') as f:
                pickle.dump(vox_dtw_dist, f)


    subjec_id=torch.tensor([int(x.subject.values) for x in voxs_dat])
    plt.imshow(vox_dtw_dist/vox_dtw_dist.max(),vmin=.7)
    _,sub_line=np.unique(subjec_id,return_index=True)
    [plt.axhline(x,color='k') for x in sub_line]
    [plt.axvline(x, color='k') for x in sub_line]
    plt.show()


    cluster=AgglomerativeClustering(n_clusters=8,affinity='precomputed',linkage='complete')
    cluster = cluster.fit(vox_dtw_dist)
    sort_idx=np.argsort(cluster.labels_)
    [subjec_id[cluster.labels_ == x] for x in np.arange(8)]
    subjec_id[sort_idx]

    a=vox_dtw_dist[sort_idx,:]
    a=a[:,sort_idx]
    plt.imshow(a,vmin=400)
    plt.show()


#     all_hamming = np.zeros((len(voxs_dat), len(voxs_dat)))
#     Y_tensor=torch.tensor([x.stimulus_id.values for x in voxs_dat])
#     subjec_id=torch.tensor([int(x.subject.values) for x in voxs_dat])
#     _,sub_line=np.unique(subjec_id,return_index=True)
#
#
#     hamming_dist=pairwise_distances(Y_tensor,metric='hamming')
#
#
#
#     plt.imshow(hamming_dist,vmin=.95)
#     [plt.axhline(x) for x in sub_line]
#     plt.show()
#     for idx,vox1 in tqdm(enumerate(voxs_dat)):
#         for idy,vox2 in enumerate(voxs_dat):
#
#             all_hamming[idx,idy]=hamming(vox1.stimulus_id.values,vox2.stimulus_id.values)
#
#     # for idx,vox1 in tqdm(enumerate(voxs_dat)):
#     #     X_tensor=torch.tensor(vox1.stimulus_id.values).unsqueeze(0).unsqueeze(0).repeat(len(voxs_dat),1,1)
#     #     Y_tensor=torch.tensor([x.stimulus_id.values for x in voxs_dat]).unsqueeze(1)
#     #     res=sdtw(X_tensor,Y_tensor)
#     #     plt.plot(res[1:])
#     #     plt.show()
#     #     for idy,vox2 in enumerate(voxs_dat):
#     #         True
#     #         X = torch.tensor(vox1.stimulus_id.values).unsqueeze(0).unsqueeze(0)
#     #         Y = torch.tensor(vox2.stimulus_id.values).unsqueeze(0).unsqueeze(0)
#     #         res=sdtw(X, Y)
#     #         all_distance[idx,idy]=res.sum()
#     #
#     # all_kendall_tau = np.zeros((len(voxs_dat), len(voxs_dat)))
#     # all_kendall_pval = np.zeros((len(voxs_dat), len(voxs_dat)))
#     # for idx, vox1 in tqdm(enumerate(voxs_dat)):
#     #     for idy, vox2 in enumerate(voxs_dat):
#     #         tau,pval=kendalltau(vox1.stimulus_id,vox2.stimulus_id)
#     #         all_kendall_tau[idx, idy] = tau
#     #         all_kendall_pval[idx, idy] = pval
#
#     alignment.plot(type='threeway')
#     plt.show()
#         fig = plt.figure(figsize=(11, 8))
#         for id, grp in enumerate(model_score.raw.groupby('subject')):
#             ax = None
#             ax = plt.subplot(3, 3, id + 1, frameon=True, sharex=ax)
#             x = grp[1].tval_lang.values.flatten()
#             y = grp[1].mean('split').values.flatten()
#             ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k", c='r')
#             b, a = np.polyfit(x, y, deg=1)
#             xseq = np.linspace(0, 1, num=100)
#             ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
#             ax.set_title(grp[0])
#         plt.tight_layout()
#         ax.set_xlabel('tval')
#         ax.set_ylabel('variance explained')
#
#         save_loc = Path(
#             f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_score_vs_tvals.png')
#         fig.savefig(save_loc.__str__(), dpi=250, format='png', metadata=None, bbox_inches=None, pad_inches=0.1,
#                     facecolor='auto', edgecolor='auto', backend=None)
#
#         width = 0.7 # the width of the bars
#         fig = plt.figure(figsize=(11, 8))
#         d = {'tval': model_score.raw.tval_lang,
#              'vox_corr': model_score.raw.repetition_corr,
#              'vox_score': model_score.raw.mean('split').values.flatten(),
#              'subject': model_score.raw.subject}
#         df1 = pd.DataFrame(d)
#
#         fig = plt.figure(figsize=(8, 11))
#         ax = plt.axes((.1, .45, .55, .45 * (8 / 11)))
#         sns.scatterplot(data=df1, x='tval', y='vox_score', hue='subject',s=25)
#         p2 = plt.axhline(y=0, color=(.5,.5,.5),zorder=0,linestyle='--')
#         xlims=ax.get_xlim()
#         ylims=ax.get_ylim()
#         regr = linear_model.LinearRegression()
#         regr.fit( np.stack([df1.tval.values,
#                            np.ones(df1.tval.shape)]).transpose(), np.expand_dims(df1.vox_score.values,axis=1))
#         tval_predict=np.arange(np.min(model_score.raw.tval_lang.values),np.max(model_score.raw.tval_lang.values),step=0.1)
#         tval_predict=np.stack([tval_predict,np.ones(tval_predict.shape)]).transpose()
#         y_pred = regr.predict(tval_predict)
#         plt.plot(tval_predict, y_pred, color="k", linewidth=3)
#         ax = plt.axes((.1, .8, .55, .15 * (8 / 11)))
#         sns.histplot(data=df1, x="tval",color='#3399ff')
#         ax.set_xlim(xlims)
#         ax.set_xlabel('')
#         ax.set_xticklabels('')
#         ax.set_title(f'model: {candidate.identifier}\n benchmark: {benchmark_id},\nregression slope:{regr.coef_[0][0]:.5f}')
#         ax = plt.axes((.68, .45, .2, .45 * (8 / 11)))
#         sns.histplot(data=df1, y="vox_score", color='#3399ff')
#         ax.set_ylabel('')
#         ax.set_yticklabels('')
#         ax.set_ylim(ylims)
#         df1_sort=df1.sort_values('tval')
#         tval=np.expand_dims(df1_sort.tval.values,axis=1)
#         vox_score_sort=np.expand_dims(df1_sort.vox_score.values,axis=1)
#         ax = plt.axes((.1, .1, .55, .35 * (8 / 11)))
#         ax.plot(np.flipud(vox_score_sort)[:1000],linewidth=.5)
#         ax.set_xlabel('Voxel ID, sorted by langloc')
#         ax.set_ylabel('Pearson correlation (vox_score)')
#         #fig.show()
#         save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_aggregate_score_vs_tvals.png')
#         fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
#         save_loc = Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/plots/results_{candidate.identifier}_{benchmark_id}_aggregate_score_vs_tvals.eps')
#         fig.savefig(save_loc.__str__(),format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
