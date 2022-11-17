import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_dataset, load_benchmark,load_model,ArtificialSubject,import_plugin, score

# load benchmark
ANNSet1=load_benchmark('ANNSet1_fMRI.train.language_top_90-linear')

# load model
candidate=load_model('distilgpt2')
model_score=ANNSet1(candidate)



fig = plt.figure(figsize=(11, 8))
for id,grp in enumerate(model_score.raw.groupby('subject')):
    ax = None
    ax = plt.subplot(3, 3, id + 1, frameon=True, sharex=ax)
    x=grp[1].S_vs_N_ratio.values.flatten()
    y=grp[1].mean('split').values.flatten()
    ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(0, 1, num=100)
    ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
    ax.set_title(grp[0])
plt.tight_layout()
ax.set_xlabel('S_vs_N ratio')
ax.set_ylabel('variance explained')
fig.show()


fig = plt.figure(figsize=(11, 8))
for id,grp in enumerate(model_score.raw.groupby('subject')):
    ax = None
    ax = plt.subplot(3, 3, id + 1, frameon=True, sharex=ax)
    x=grp[1].repetition_corr.values.flatten()
    y=grp[1].mean('split').values.flatten()
    ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k",c='r')
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(0, 1, num=100)
    ax.plot(np.sort(x), a + b * np.sort(x), color="k", lw=2.5)
    ax.set_title(grp[0])
plt.tight_layout()
ax.set_xlabel('repetition correlatione')
ax.set_ylabel('variance explained')
fig.show()
