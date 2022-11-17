import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle

# load benchmark
ANNSet1=load_benchmark('ANNSet1_fMRI.train.language_top_90-linear')
Pereira234=load_benchmark('Pereira2018.243sentences-linear')
Pereira384=load_benchmark('Pereira2018.384sentences-linear')


# load model
# 1.
candidate=load_model('roberta-base')
roberta_score=ANNSet1(candidate)
roberta_score_p234=Pereira234(candidate)
roberta_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/roberta_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(roberta_score, f)


save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/roberta_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(roberta_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/roberta_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(roberta_score_p384, f)


# 2.
candidate=load_model('xlm-mlm-en-2048')
xlm_score=ANNSet1(candidate)
xlm_score_p234=Pereira234(candidate)
xlm_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlm_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlm_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlm_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlm_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlm_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlm_score_p384, f)

# 3.
candidate=load_model('xlnet-large-cased')
xlnet_score=ANNSet1(candidate)
xlnet_score_p234=Pereira234(candidate)
xlnet_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlnet_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlnet_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlnet_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlnet_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/xlnet_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(xlnet_score_p384, f)

# 4.
candidate=load_model('albert-xxlarge-v2')
albert_score=ANNSet1(candidate)
albert_score_p234=Pereira234(candidate)
albert_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/albert_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(albert_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/albert_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(albert_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/albert_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(albert_score_p384, f)

# 5.
candidate=load_model('bert-base-uncased')
bert_score=ANNSet1(candidate)
bert_score_p234=Pereira234(candidate)
bert_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/bert_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(bert_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/bert_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(bert_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/bert_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(bert_score_p384, f)

# 6.
candidate=load_model('gpt2-xl')
gpt2_score=ANNSet1(candidate)
gpt2_score_p234=Pereira234(candidate)
gpt2_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/gpt2_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(gpt2_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/gpt2_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(gpt2_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/gpt2_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(gpt2_score_p384, f)

# 7.
candidate=load_model('ctrl')
ctrl_score=ANNSet1(candidate)
ctrl_score_p234=Pereira234(candidate)
ctrl_score_p384=Pereira384(candidate)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ctrl_score_ANN.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(ctrl_score, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ctrl_score_p234.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(ctrl_score_p234, f)

save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ctrl_score_p384.pkl')
with open(save_dir.__str__(), 'wb') as f:
    pickle.dump(ctrl_score_p384, f)

#

fig = plt.figure(figsize=(11, 8))
ax = plt.axes((.1, .4, .35, .35))
x=roberta_score.raw.S_vs_N_ratio.values.flatten()
y=roberta_score.raw.mean('split').values.flatten()
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(x, y, deg=1)
xseq = np.linspace(0, 1, num=100)
ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
ax.set_title('all voxels')
plt.tight_layout()
ax.set_xlabel('S_vs_N ratio')
ax.set_ylabel('variance explained')

ax = plt.axes((.6, .4, .35, .35))
x=roberta_score.raw.repetition_corr.values.flatten()
y=roberta_score.raw.mean('split').values.flatten()
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k",c='r')
b, a = np.polyfit(x, y, deg=1)
xseq = np.linspace(0, 1, num=100)
ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
ax.set_title('all voxels')
plt.tight_layout()
ax.set_xlabel('repetition correlatione')
ax.set_ylabel('variance explained')
fig.show()


def plot_scores_against_repetition_ratio(model_score,save_name):
    fig = plt.figure(figsize=(11, 8))
    for id, grp in enumerate(model_score.raw.groupby('subject')):
        ax = None
        ax = plt.subplot(3, 3, id + 1, frameon=True, sharex=ax)
        x = grp[1].repetition_corr.values.flatten()
        y = grp[1].mean('split').values.flatten()
        ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k", c='r')
        b, a = np.polyfit(x, y, deg=1)
        xseq = np.linspace(0, 1, num=100)
        ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
        ax.set_title(grp[0])
    plt.tight_layout()
    ax.set_xlabel('repetition correlatione')
    ax.set_ylabel('variance explained')
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{save_name}_vs_rep_corr.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{save_name}_vs_rep_corr.eps')
    fig.savefig(save_loc.__str__(), format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    x = model_score.raw.S_vs_N_ratio.values.flatten()
    y = model_score.raw.mean('split').values.flatten()
    ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(0, 1, num=100)
    ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
    ax.set_title(f'{save_name} all voxels')
    plt.tight_layout()
    ax.set_xlabel('S_vs_N ratio')
    ax.set_ylabel('variance explained')
    ax = plt.axes((.6, .4, .35, .35))
    x = model_score.raw.repetition_corr.values.flatten()
    y = model_score.raw.mean('split').values.flatten()
    ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k", c='r')
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(0, 1, num=100)
    ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
    ax.set_title(f'{save_name} all voxels')
    plt.tight_layout()
    ax.set_xlabel('repetition correlatione')
    ax.set_ylabel('variance explained')
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{save_name}_vs_rep_corr_all_subs.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{save_name}_vs_rep_corr_subs.eps')
    fig.savefig(save_loc.__str__(), format='eps',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

if __name__ == '__main__':
    plot_scores_against_repetition_ratio(roberta_score, 'roberta-base')
    plot_scores_against_repetition_ratio(xlm_score,'xlm-mlm-en-2048')
    plot_scores_against_repetition_ratio(xlnet_score, 'xlnet-large-cased')
    plot_scores_against_repetition_ratio(albert_score, 'albert-xxlarge-v2')
    plot_scores_against_repetition_ratio(bert_score, 'bert-base-uncased')
    plot_scores_against_repetition_ratio(gpt2_score, 'gpt2-xl')
    plot_scores_against_repetition_ratio(ctrl_score, 'ctrl')
    #
    model_Pereira=[ np.mean([roberta_score_p234.values,roberta_score_p384.values]),
                    np.mean([xlm_score_p234.values, xlm_score_p234.values]),
                    np.mean([xlnet_score_p234.values, xlnet_score_p384.values]),
                    np.mean([albert_score_p234.values, albert_score_p234.values]),
                    np.mean([bert_score_p234.values, bert_score_p384.values]),
                    np.mean([gpt2_score_p234.values, gpt2_score_p384.values]),
                    np.mean([ctrl_score_p234.values, ctrl_score_p384.values])]

    model_ANNSet1=[ roberta_score.values[0],
                    xlm_score.values[0],
                    xlnet_score.values[0],
                    albert_score.values[0],
                    bert_score.values[0],
                    gpt2_score.values[0],
                    ctrl_score.values[0]]
    model_ANNSet1_err=[ roberta_score.values[1],
                    xlm_score.values[1],
                    xlnet_score.values[1],
                    albert_score.values[1],
                    bert_score.values[1],
                    gpt2_score.values[1],
                    ctrl_score.values[1]]


    labels = ['roberta-base','xlm-mlm-en-2048','xlnet-large-cased','albert-xxlarge-v2','bert-base-uncased','gpt2-xl','ctrl']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes((.1, .4, .35, .35))
    rects1 = ax.bar(x - width / 2, np.stack(model_Pereira), width, color=np.divide((55, 76, 128), 256),label='Pereira')
    rects2 = ax.bar(x + width / 2, np.stack(model_ANNSet1).squeeze(), width, label='ANNSet1_fMRI',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, np.stack(model_ANNSet1).squeeze(),
                yerr=np.stack(model_ANNSet1_err).squeeze(), linestyle='',
                color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title('Comparison of model performance between Pereira and ANNSet1 fMRI')
    ax.set_xticks(x, labels)
    ax.set_ylim((-.1, 1.1))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson corr')
    #fig.tight_layout()

    fig.show()
    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/ANNSet1_vs_Pereira_performance_all_models.png')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    save_loc = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/ANNSet1_vs_Pereira_performance_all_models.eps')
    fig.savefig(save_loc.__str__(), dpi=250,format='png',metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
