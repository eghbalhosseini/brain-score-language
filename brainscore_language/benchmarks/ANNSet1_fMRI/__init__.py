from brainscore_language import benchmark_registry
from .benchmark import ANNSet1_fMRI_lang_top_90

benchmark_registry['ANNSet1_fMRI.train.language_top_90-linear'] = ANNSet1_fMRI_lang_top_90

#benchmark_registry['ANNSet1_fMRI.train.language_top_70-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.language_top_70')
#benchmark_registry['ANNSet1_fMRI.train.language_top_80-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.language_top_80')
#benchmark_registry['ANNSet1_fMRI.train.auditory-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.auditory')
#benchmark_registry['ANNSet1_fMRI.train.visual-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.visual')

