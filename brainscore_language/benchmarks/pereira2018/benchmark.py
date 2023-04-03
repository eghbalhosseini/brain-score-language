import logging
import xarray as xr
import numpy as np
import re
from scipy.stats import median_abs_deviation
import pandas as pd
from pathlib import Path
from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
import torch
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

BIBTEX = """@article{pereira2018toward,
  title={Toward a universal decoder of linguistic meaning from brain activation},
  author={Pereira, Francisco and Lou, Bin and Pritchett, Brianna and Ritter, Samuel and Gershman, Samuel J 
          and Kanwisher, Nancy and Botvinick, Matthew and Fedorenko, Evelina},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={1--13},
  year={2018},
  publisher={Nature Publishing Group}
}"""


def Pereira2018_243sentences():
    return _Pereira2018ExperimentLinear(experiment='243sentences', ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))

def Pereira2018_243sentences_ds_max(samples=100):
    return _Pereira2018ExperimentSamplerLinear(experiment='243sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/ds_max_Ns_{samples}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_243sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))
def Pereira2018_243sentences_ds_max_rand(samples=100,rand_id=1):
    return _Pereira2018ExperimentSamplerLinear(experiment='243sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/ds_max_Ns_{samples}_rand_{rand_id}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_243sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))

def Pereira2018_243sentences_ds_min(samples=100):
    return _Pereira2018ExperimentSamplerLinear(experiment='243sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/2-ds_max_Ns_{samples}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_243sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))

def Pereira2018_243sentences_ds_min_rand(samples=100,rand_id=1):
    return _Pereira2018ExperimentSamplerLinear(experiment='243sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/2-ds_max_Ns_{samples}_rand_{rand_id}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_243sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='CHl_9aFHIWVnPW_njePfy28yzggKuUPw',
        sha1='5e23de899883828f9c886aec304bc5aa0f58f66c',
        raw_kwargs=dict(
            version_id='uZye03ENmn.vKB5mARUGhcIY_DjShtPD',
            sha1='525a6ac8c14ad826c63fdd71faeefb8ba542d5ac',
            raw_kwargs=dict(
                version_id='XVTo58Po5YrNjTuDIWrmfHI0nbN2MVZa',
                sha1='34ba453dc7e8a19aed18cc9bca160e97b4a80be5'
            )
        )
    ))


def Pereira2018_384sentences():
    return _Pereira2018ExperimentLinear(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
   ))

def Pereira2018_384sentences_ds_max(samples=100):
    return _Pereira2018ExperimentSamplerLinear(experiment='384sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/ds_max_Ns_{samples}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_384sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    )
    )

def Pereira2018_384sentences_ds_max_rand(samples=100,rand_id=1):
    return _Pereira2018ExperimentSamplerLinear(experiment='384sentences',
                                               sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/ds_max_Ns_{samples}_rand_{rand_id}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_384sentences.pkl',
                                               ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    )
    )

def Pereira2018_384sentences_ds_min(samples=100):
    return _Pereira2018ExperimentSamplerLinear(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    ),
    sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/2-ds_max_Ns_{samples}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_384sentences.pkl')

def Pereira2018_384sentences_ds_min_rand(samples=100,rand_id=1):
    return _Pereira2018ExperimentSamplerLinear(experiment='384sentences', ceiling_s3_kwargs=dict(
        version_id='sjlnXr5wXUoGv6exoWu06C4kYI0KpZLk',
        sha1='fc895adc52fd79cea3040961d65d8f736a9d3e29',
        raw_kwargs=dict(
            version_id='Hi74r9UKfpK0h0Bjf5DL.JgflGoaknrA',
            sha1='ce2044a7713426870a44131a99bfc63d8843dae0',
            raw_kwargs=dict(
                version_id='m4dq_ouKWZkYtdyNPMSP0p6rqb7wcYpi',
                sha1='fe9fb24b34fd5602e18e34006ac5ccc7d4c825b8'
            )
        )
    ),
    sampler=f'/om2/user/ehoseini/MyData/brain-score-language/output/2-ds_max_Ns_{samples}_rand_{rand_id}_corrcoef_roberta-base_xlm-mlm-en-2048_xlnet-large-cased_albert-xxlarge-v2_bert-large-uncased-whole-word-masking_gpt2-xl_ctrl_Pereira2018_language_384sentences.pkl')


class _Pereira2018ExperimentLinear(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in the human language system in response to natural sentences,
    recorded by Pereira et al. 2018.
    Alignment of neural activity between model and human subjects is evaluated via cross-validated linear predictivity.

    This benchmark builds off the behavioral benchmark introduced
    in Schrimpf et al. 2021 (https://www.pnas.org/doi/10.1073/pnas.2105646118), but:

    * computes neural alignment to each of the two experiments ({243,384}sentences) separately, as well as ceilings
    * requires the model to have committed to neural readouts (e.g. layer 41 corresponds to the language system),
        rather than testing every layer separately

    Each of these benchmarks evaluates one of the two experiments, the overall Pereira2018-linear score is the mean of
    the two ceiling-normalized scores.
    """

    def __init__(self, experiment: str, ceiling_s3_kwargs: dict):
        self.data = self._load_data(experiment)
        self.metric = load_metric('linear_pearsonr')
        identifier = f'Pereira2018.{experiment}-linear'
        ceiling = self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_Pereira2018ExperimentLinear, self).__init__(
            identifier=identifier,
            version=1,
            parent='Pereira2018-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        return data

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id, sha1=sha1)
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
            ceiling.attrs['raw'] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        passages = self.data['passage_label'].values
        predictions = []
        for passage in sorted(set(passages)):  # go over individual passages, sorting to keep consistency across runs
            passage_indexer = [stimulus_passage == passage for stimulus_passage in passages]
            passage_stimuli = stimuli[passage_indexer]
            passage_predictions = candidate.digest_text(passage_stimuli.values)['neural']
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_id'].values
            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        # layerwise scores
        raw_scores = []
        raw_raw_scores=[]
        for layer_id, prediction in predictions.groupby('layer'):
            raw_score = self.metric(prediction, self.data)
            raw_scores.append(raw_score)
            raw_raw_scores.append(raw_score.raw.expand_dims(dim={"layer": [layer_id]}, axis=0))
            #raw_scores.append(raw_score.raw.expand_dims(dim={"layer": [layer_id]}, axis=0))
        raw_scores = xr.concat(raw_scores, dim='layer')
        raw_raw_scores=xr.concat(raw_raw_scores, dim='layer')
        scores=[]
        for l,sc in raw_scores.groupby('layer'):
            sc = ceiling_normalize(sc, self.ceiling)
            scores.append(sc)
        scores=xr.concat(scores,dim='layer')
        score = Score([scores.values], coords={'aggregation': ['center'],
                                               'layer':raw_scores.layer.values},
                      dims=['aggregation','layer'])
        score.attrs['raw'] = raw_scores
        score.attrs['raw_raw']=raw_raw_scores
        #score = raw_scores.mean('split')
        #score = score.groupby('subject').median()
        #center = score.median('subject')
        #subject_values = np.nan_to_num(score.values, nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        #subject_axis = score.dims.index(score['subject'].dims[0])
        #error = median_abs_deviation(subject_values, axis=subject_axis)
        # ceiling normalize
        #scores=[]
        #for l,sc in center.groupby('layer'):
        #    sc = ceiling_normalize(sc, self.ceiling)
        #    scores.append(sc)
        #scores=xr.concat(scores,dim='layer')

        #score = Score([scores.values, error], coords={'aggregation': ['center', 'error'],
        #                                       'layer':scores.layer.values},
        #              dims=['aggregation','layer'])
        #score.attrs['raw'] = raw_scores
        #score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
        #                             "then median of subject scores"
        return score

class _Pereira2018ExperimentSamplerLinear(_Pereira2018ExperimentLinear):
    def __init__(self,experiment: str,sampler:str,ceiling_s3_kwargs: dict):

        self.sampler=sampler
        self.data = self._load_data(experiment)
        self.metric = load_metric('linear_pearsonr')
        data_id=self.data.attrs['identifier']
        base_identifier = f'Pereira2018.{experiment}-linear'
        identifier = f'Pereira2018.{experiment}-sampler{data_id}-linear'
        ceiling = self._load_ceiling(identifier=base_identifier, **ceiling_s3_kwargs)
        super(_Pereira2018ExperimentSamplerLinear, self).__init__(
            experiment=experiment, ceiling_s3_kwargs=ceiling_s3_kwargs)

    def _load_data(self, experiment: str) -> NeuroidAssembly:
        data = load_dataset('Pereira2018.language')
        sampler_dat=pd.read_pickle(self.sampler)
        sample_stimulus_id=sampler_dat['coordinates']['stimulus_id']
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't

        sample_loc=np.sum([data.stimulus_id.values==x for x in sample_stimulus_id],axis=0).astype(bool)
        sampler_dat['coordinates']['stimulus']
        data_sample=data.sel(presentation=sample_loc)
        assert len([list(data_sample.stimulus.values).index(x) for x in sampler_dat['coordinates']['stimulus']])==len(sample_stimulus_id)
        # manually get the sampler identifier
        filename=Path(self.sampler).name
        res = re.search('roberta',filename)
        filename[:res.start()-1]
        data_sample.attrs['identifier'] = f"{data.identifier}.{experiment}.{filename[:res.start()-1]}"
        return data_sample

