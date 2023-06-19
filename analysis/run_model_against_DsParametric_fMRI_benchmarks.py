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
    models = ['roberta-base',
              'xlnet-large-cased',
              'bert-large-uncased-whole-word-masking',
              'xlm-mlm-en-2048',
              'gpt2-xl',
              'albert-xxlarge-v2',
              'ctrl']
    model = models[model_id]
    candidate = load_model(model)
    DsParametric_benchmark_set=['DsParametric_fmri.max.language_top_90-linear_pearsonr',
                          'DsParametric_fmri.max.language_top_80-linear_pearsonr',
                            'DsParametric_fmri.min.language_top_90-linear_pearsonr',
                            'DsParametric_fmri.min.language_top_80-linear_pearsonr',
                            'DsParametric_fmri.random.language_top_90-linear_pearsonr',
                            'DsParametric_fmri.random.language_top_80-linear_pearsonr',

                                'DsParametric_fmri.max.language_top_90-rgcv_linear_pearsonr',
                                'DsParametric_fmri.max.language_top_80-rgcv_linear_pearsonr',
                                'DsParametric_fmri.min.language_top_90-rgcv_linear_pearsonr',
                                'DsParametric_fmri.min.language_top_80-rgcv_linear_pearsonr',
                                'DsParametric_fmri.random.language_top_90-rgcv_linear_pearsonr',
                                'DsParametric_fmri.random.language_top_80-rgcv_linear_pearsonr',
                            ]
    for ann_benchmark in DsParametric_benchmark_set:
        benchmark=load_benchmark(ann_benchmark)
        score=benchmark(candidate)
        bench_id=benchmark.identifier.replace('.','-')

        save_dir = Path(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/score/score_{candidate.identifier}_{benchmark.identifier}.pkl')
        with open(save_dir.__str__(), 'wb') as f:
            pickle.dump(score, f)

