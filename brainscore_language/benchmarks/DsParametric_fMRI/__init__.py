from brainscore_language import benchmark_registry
from brainscore_language.benchmarks.DsParametric_fMRI.benchmark import DsParametric_fMRI

# max
benchmark_registry['DsParametric_fmri.max.language_top_90-linear'] = lambda: DsParametric_fMRI('language_top_90',None,'max')
benchmark_registry['DsParametric_fmri.max.language_top_80-linear'] = lambda: DsParametric_fMRI('language_top_80',None,'max')
benchmark_registry['DsParametric_fmri.max.language_top_70-linear'] = lambda: DsParametric_fMRI('language_top_70',None,'max')

# min
benchmark_registry['DsParametric_fmri.min.language_top_90-linear'] = lambda: DsParametric_fMRI('language_top_90',None,'min')
benchmark_registry['DsParametric_fmri.min.language_top_80-linear'] = lambda: DsParametric_fMRI('language_top_80',None,'min')
benchmark_registry['DsParametric_fmri.min.language_top_70-linear'] = lambda: DsParametric_fMRI('language_top_70',None,'min')

# random
benchmark_registry['DsParametric_fmri.random.language_top_90-linear'] = lambda: DsParametric_fMRI('language_top_90',None,'random')
benchmark_registry['DsParametric_fmri.random.language_top_80-linear'] = lambda: DsParametric_fMRI('language_top_80',None,'random')
benchmark_registry['DsParametric_fmri.random.language_top_70-linear'] = lambda: DsParametric_fMRI('language_top_70',None,'random')

