import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle

# load benchmark
ANNSet1_90=load_benchmark('ANNSet1_fMRI.train.language_top_90-linear')
ANNSet1_80=load_benchmark('ANNSet1_fMRI.train.language_top_80-linear')
ANNSet1_70=load_benchmark('ANNSet1_fMRI.train.language_top_70-linear')

ANNSet1_aud=load_benchmark('ANNSet1_fMRI.train.auditory-linear')
ANNSet1_vis=load_benchmark('ANNSet1_fMRI.train.visual-linear')