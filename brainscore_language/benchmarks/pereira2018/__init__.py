from brainscore_language import benchmark_registry
from .benchmark import Pereira2018_243sentences, Pereira2018_384sentences, Pereira2018_243sentences_ds_max,Pereira2018_243sentences_ds_min,Pereira2018_384sentences_ds_min,Pereira2018_384sentences_ds_max

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences

benchmark_registry['Pereira2018.243sentences.ds.max-linear'] = Pereira2018_243sentences_ds_max
benchmark_registry['Pereira2018.384sentences.ds.max-linear'] = Pereira2018_384sentences_ds_max

benchmark_registry['Pereira2018.243sentences.ds.min-linear'] = Pereira2018_243sentences_ds_min
benchmark_registry['Pereira2018.384sentences.ds.min-linear'] = Pereira2018_384sentences_ds_min
