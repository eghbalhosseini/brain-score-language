import numpy as np
import matplotlib.pyplot as plt
from brainscore_language import load_benchmark,load_model
from pathlib import Path
import pickle
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser(description='run model on all Pereira benchmarks')
parser.add_argument('model_id', type=str)
args=parser.parse_args()
if __name__ == '__main__':
    model_id = int(args.model_id)


    # for all models compute benchmarks:
    models = ['distilgpt2-layerwise', 'gpt2-layerwise', 'gpt2-medium-layerwise', 'gpt2-large-layerwise']
    model = models[model_id]
    candidate = load_model(model)
    ann_benchmark_set=['ANNSet1_fMRI.train.language_top_90-linear',
                       'ANNSet1_fMRI.train.auditory-linear',
                       'ANNSet1_fMRI.train.visual-linear',
                       'ANNSet1_fMRI.best.language_top_90_V2-linear',
                       'ANNSet1_fMRI_WOPeriod.train.language_top_90-linear']

    for ann_benchmark in ann_benchmark_set:
        benchmark=load_benchmark(ann_benchmark)
        score=benchmark(candidate)
        bench_id=benchmark.identifier.replace('.','-')
        save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/score_{candidate.identifier}_{benchmark.identifier}.pkl')
        with open(save_dir.__str__(), 'wb') as f:
            pickle.dump(score, f)

