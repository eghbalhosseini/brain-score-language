from brainscore_language import benchmark_registry
from .benchmark import Pereira2018_243sentences, Pereira2018_384sentences, Pereira2018_243sentences_ds_max,Pereira2018_243sentences_ds_min,Pereira2018_384sentences_ds_min,Pereira2018_384sentences_ds_max
from .benchmark import Pereira2018_243sentences_ds_max_rand , Pereira2018_243sentences_ds_min_rand,Pereira2018_384sentences_ds_min_rand ,Pereira2018_384sentences_ds_max_rand

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences


# Pereira2018 384 samples
# max
benchmark_registry['Pereira2018.384sentences.ds.max.100-linear'] = lambda :Pereira2018_384sentences_ds_max(samples=100)
benchmark_registry['Pereira2018.384sentences.ds.max.150-linear'] = lambda :Pereira2018_384sentences_ds_max(samples=150)
benchmark_registry['Pereira2018.384sentences.ds.max.200-linear'] = lambda :Pereira2018_384sentences_ds_max(samples=200)
benchmark_registry['Pereira2018.384sentences.ds.max.250-linear'] = lambda :Pereira2018_384sentences_ds_max(samples=250)
# max rand
benchmark_registry['Pereira2018.384sentences.ds.max.100.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_max_rand(samples=100,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.max.150.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_max_rand(samples=150,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.max.200.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_max_rand(samples=200,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.max.250.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_max_rand(samples=250,rand_id=0)
# min
benchmark_registry['Pereira2018.384sentences.ds.min.100-linear'] = lambda :Pereira2018_384sentences_ds_min(samples=100)
benchmark_registry['Pereira2018.384sentences.ds.min.150-linear'] = lambda :Pereira2018_384sentences_ds_min(samples=150)
benchmark_registry['Pereira2018.384sentences.ds.min.200-linear'] = lambda :Pereira2018_384sentences_ds_min(samples=200)
benchmark_registry['Pereira2018.384sentences.ds.min.250-linear'] = lambda :Pereira2018_384sentences_ds_min(samples=250)
# min rand
benchmark_registry['Pereira2018.384sentences.ds.min.100.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_min_rand(samples=100,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.min.150.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_min_rand(samples=150,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.min.200.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_min_rand(samples=200,rand_id=0)
benchmark_registry['Pereira2018.384sentences.ds.min.250.rand.0-linear'] = lambda :Pereira2018_384sentences_ds_min_rand(samples=250,rand_id=0)

#Pereira2018 243 samples
# min
benchmark_registry['Pereira2018.243sentences.ds.min.100-linear'] = lambda :Pereira2018_243sentences_ds_min(samples=100)
benchmark_registry['Pereira2018.243sentences.ds.min.150-linear'] = lambda :Pereira2018_243sentences_ds_min(samples=150)
benchmark_registry['Pereira2018.243sentences.ds.min.200-linear'] = lambda :Pereira2018_243sentences_ds_min(samples=200)
# min rand
benchmark_registry['Pereira2018.243sentences.ds.min.100.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_min_rand(samples=100,rand_id=0)
benchmark_registry['Pereira2018.243sentences.ds.min.150.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_min_rand(samples=150,rand_id=0)
benchmark_registry['Pereira2018.243sentences.ds.min.200.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_min_rand(samples=200,rand_id=0)
# max
benchmark_registry['Pereira2018.243sentences.ds.max.100-linear'] = lambda :Pereira2018_243sentences_ds_max(samples=100)
benchmark_registry['Pereira2018.243sentences.ds.max.150-linear'] = lambda :Pereira2018_243sentences_ds_max(samples=150)
benchmark_registry['Pereira2018.243sentences.ds.max.200-linear'] = lambda :Pereira2018_243sentences_ds_max(samples=200)
# max rand
benchmark_registry['Pereira2018.243sentences.ds.max.100.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_max_rand(samples=100,rand_id=0)
benchmark_registry['Pereira2018.243sentences.ds.max.150.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_max_rand(samples=150,rand_id=0)
benchmark_registry['Pereira2018.243sentences.ds.max.200.rand.0-linear'] = lambda :Pereira2018_243sentences_ds_max_rand(samples=200,rand_id=0)




