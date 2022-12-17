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

    Pereria384_ds_max=[load_benchmark(f'Pereira2018.384sentences.ds.max.{x}-linear') for x in [150,200]]
    Pereria384_ds_max_rand = [load_benchmark(f'Pereira2018.384sentences.ds.max.{x}.rand.0-linear') for x in [150, 200]]
    Pereria384_ds_min = [load_benchmark(f'Pereira2018.384sentences.ds.min.{x}-linear') for x in [150, 200]]
    Pereria384_ds_min_rand = [load_benchmark(f'Pereira2018.384sentences.ds.min.{x}.rand.0-linear') for x in [150, 200]]

    Pereria243_ds_max = [load_benchmark(f'Pereira2018.243sentences.ds.max.{x}-linear') for x in [150, 200]]
    Pereria243_ds_max_rand = [load_benchmark(f'Pereira2018.243sentences.ds.max.{x}.rand.0-linear') for x in [150, 200]]
    Pereria243_ds_min = [load_benchmark(f'Pereira2018.243sentences.ds.min.{x}-linear') for x in [150, 200]]
    Pereria243_ds_min_rand = [load_benchmark(f'Pereira2018.243sentences.ds.min.{x}.rand.0-linear') for x in [150, 200]]


    # for all models compute benchmarks:
    models = ['roberta-base', 'xlm-mlm-en-2048', 'xlnet-large-cased', 'albert-xxlarge-v2', 'bert-base-uncased',
              'gpt2-xl', 'ctrl']

    model_scores_pereira_384_ds_max=[]
    model_scores_pereira_384_ds_max_rand=[]
    model_scores_pereira_384_ds_min=[]
    model_scores_pereira_384_ds_min_rand=[]

    model_scores_pereira_243_ds_max=[]
    model_scores_pereira_343_ds_max_rand=[]
    model_scores_pereira_243_ds_min=[]
    model_scores_pereira_243_ds_min_rand=[]

    model=models[model_id]

    candidate=load_model(model)
    model_score384_ds_max=[x(candidate) for x in Pereria384_ds_max]
    model_score384_ds_max_rand = [x(candidate) for x in Pereria384_ds_max_rand]
    model_score384_ds_min = [x(candidate) for x in Pereria384_ds_min]
    model_score384_ds_min_rand = [x(candidate) for x in Pereria384_ds_min_rand]

    model_score243_ds_max = [x(candidate) for x in Pereria243_ds_max]
    model_score243_ds_max_rand = [x(candidate) for x in Pereria243_ds_max_rand]
    model_score243_ds_min = [x(candidate) for x in Pereria243_ds_min]
    model_score243_ds_min_rand = [x(candidate) for x in Pereria243_ds_min_rand]


    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score384_ds_max, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_max_rand.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score384_ds_max_rand, f)


    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score384_ds_min, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score384_ds_min_rand.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score384_ds_min_rand, f)
    ##

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_max.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score243_ds_max, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_max_rand.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score243_ds_max_rand, f)


    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_min.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score243_ds_min, f)

    save_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/{model}_score243_ds_min_rand.pkl')
    with open(save_dir.__str__(), 'wb') as f:
        pickle.dump(model_score243_ds_min_rand, f)

