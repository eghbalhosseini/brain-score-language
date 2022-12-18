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
import rsatoolbox.data as rsd
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BIBTEX = """
}"""

def Pereira2018_rsa_243sentences():
    return _Pereira2018ExperimentRSA(experiment='243sentences', ceiling_s3_kwargs=None)



class _Pereira2018ExperimentRSA(BenchmarkBase):
    def __init__(self, experiment:str,ceiling_s3_kwargs: dict ):
        self.data,self.rsa_data = self._load_data(experiment)
        self.metric = load_metric('rsa_correlation')
        identifier = f'Pereira2018.{experiment}-rsa'
        ceiling = None if not ceiling_s3_kwargs else self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_Pereira2018ExperimentRSA, self).__init__(
            identifier=identifier,
            version=1,
            parent='Pereira2018-rsa',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):

        NotImplementedError

    def _load_data(self, experiment: str):
        data = load_dataset('Pereira2018.language')
        data = data.sel(experiment=experiment)  # filter experiment
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data = data.sortby('stimulus_id')
        data.attrs['identifier'] = f"{data.identifier}.{experiment}"
        # make an RSA dataset:
        # seperate the data based on subject:
        rsd_xr=data.sortby('stimulus_num')
        rsa_dataset=[]
        for sub_id, sub_dat in rsd_xr.groupby('subject'):
            descriptors={'subject':sub_dat.subject.values[0],
                         'day':1
                         }
            ch_descriptors={'neuroid':sub_dat.neuroid_id.values,
                            'roi':sub_dat.roi.values}

            obs_descriptors={'stimulus_name':sub_dat.stimulus_id.values,
                             'stimulus_id':sub_dat.stimulus_num.values,
                             'sentence':sub_dat.sentence.values,
                             'stimulus':sub_dat.stimulus.values,
                             'session':sub_dat.stimulus_num.values*0+1,
                             }
            rsa_dataset.append(rsd.Dataset(measurements=sub_dat.values, descriptors=descriptors, obs_descriptors=obs_descriptors,
                                           channel_descriptors=ch_descriptors))
        return data, rsa_dataset

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
            passage_predictions['stimulus_id'] = 'presentation', passage_stimuli['stimulus_num'].values

            predictions.append(passage_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        predictions=predictions.sortby('stimulus_id')

        # make model data into rdm
        descriptors = {'model': candidate.identifier,
                       'Target': candidate.region_layer_mapping
                       }
        obs_descriptors = {'stimulus_id': predictions.stimulus_id.values,
                           'stimulus': predictions.stimulus.values}
        ch_descriptors = {'neuroid': predictions.neuroid_id.values,
                          'neuron_number_in_layer': predictions.neuron_number_in_layer.values}
        predictions_rsd=rsd.Dataset(predictions.values, descriptors=descriptors,
                    obs_descriptors=obs_descriptors,channel_descriptors=ch_descriptors)

        raw_score = self.metric(predictions_rsd, self.rsa_data)
        score = Score([raw_score.get_means(), raw_score.get_errorbars()[0]], coords={'aggregation': ['center', 'error'],},
                       dims=['aggregation','layer'])
        score.attrs['raw'] = raw_score
        score.attrs['noise_ceiling']=raw_score.get_noise_ceil()

        return score