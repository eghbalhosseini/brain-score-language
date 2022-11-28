from brainscore_language import benchmark_registry
from .benchmark import ANNSet1_fMRI_lang_top_90,ANNSet1_fMRI_lang_top_80,ANNSet1_fMRI_lang_top_70, ANNSet1_fMRI_auditory, ANNSet1_fMRI_visual,ANNSet1_fMRI_lang_top_90_V2

benchmark_registry['ANNSet1_fMRI.train.language_top_90-linear'] = ANNSet1_fMRI_lang_top_90
benchmark_registry['ANNSet1_fMRI.train.language_top_80-linear'] = ANNSet1_fMRI_lang_top_80
benchmark_registry['ANNSet1_fMRI.train.language_top_70-linear'] = ANNSet1_fMRI_lang_top_70
benchmark_registry['ANNSet1_fMRI.train.auditory-linear'] = ANNSet1_fMRI_auditory
benchmark_registry['ANNSet1_fMRI.train.visual-linear'] = ANNSet1_fMRI_visual



benchmark_registry['ANNSet1_fMRI.best.language_top_90_V2-linear'] = ANNSet1_fMRI_lang_top_90_V2

#benchmark_registry['ANNSet1_fMRI.train.language_top_70-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.language_top_70')
#benchmark_registry['ANNSet1_fMRI.train.language_top_80-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.language_top_80')
#benchmark_registry['ANNSet1_fMRI.train.auditory-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.auditory')
#benchmark_registry['ANNSet1_fMRI.train.visual-linear'] = lambda: ANNSet1_fMRI_benchmarkLinear(atlas='train.visual')


