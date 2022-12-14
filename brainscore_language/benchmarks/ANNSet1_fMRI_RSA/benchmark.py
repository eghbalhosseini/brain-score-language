import logging

import numpy as np
import pandas as pd
from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from scipy.stats import median_abs_deviation
from tqdm import tqdm
import xarray as xr
import rsatoolbox.data as rsd

logger = logging.getLogger(__name__)

BIBTEX = """
}"""

def ANNSet1_fMRI_RSA_lang_top_90():
    return _ANNSet1_fMRI_ExperimentRSA(atlas='train.language_top_90',ceiling_s3_kwargs=None)



class _ANNSet1_fMRI_ExperimentRSA(BenchmarkBase):
    def __init__(self, atlas:str,ceiling_s3_kwargs: dict ):
        self.data = self._load_data(atlas)
        self.metric = load_metric('rsa_correlation')
        identifier = f'ANNSet1_fMRI.{atlas}-rsa'
        ceiling = None if not ceiling_s3_kwargs else self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_ANNSet1_fMRI_ExperimentRSA, self).__init__(
            identifier=identifier,
            version=1,
            parent='ANNSet1_fMRI-rsa',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, atlas: str):
        data = load_dataset(f'ANNSet1_fMRI_RSA.{atlas}')
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data = data.sortby('stimulus_id')
        data.attrs['identifier'] = f'ANNSet1_fMRI_RSA.{atlas}'
        # make an RSA dataset:
        rep_list=[]
        for rep_id,rep in data.groupby('repeat'):
            rep_list.append(rep.assign_coords({'repeat_id':('neuroid',(np.ones(rep.shape[1]).astype(int)*rep.repeat.values))}))
        rep_xr=xr.concat(rep_list,'presentation')
        # seperate the data based on subject:
        rsa_dataset=[]
        for sub_id, sub_dat in rep_xr.groupby('subject'):
            descriptors={'subject':sub_dat.subject.values[0],
                         'day':sub_dat.sess_id.values[0]
                         }
            ch_descriptors={'neuroid':sub_dat.neuroid_id.values,
                            'roi':sub_dat.roi.values}

            obs_descriptors={'stimulus_id':sub_dat.stimulus_id.values,
                             'stimulus_num':sub_dat.stimulus_num.values,
                             'sentence':sub_dat.sentence.values,
                             'stimulus':sub_dat.stimulus.values,
                             'session':sub_dat.repeat_id.values[:,0],
                             }
            rsa_dataset.append(rsd.Dataset(measurements=sub_dat.values, descriptors=descriptors, obs_descriptors=obs_descriptors,
                                           channel_descriptors=ch_descriptors))

        # for _,sess_dat in sub_dat.groupby('repeat_id'):
            #     True
            #     descriptors={'subject':sess_dat.subject.values[0],
            #              'session':sess_dat.repeat_id.values[0],
            #              'day':sess_dat.sess_id.values[0]
            #              }
            #     ch_descriptors={'neuroid':sess_dat.neuroid_id.values,
            #                 'roi':sess_dat.roi.values}
            #
            #     obs_descriptors={'stimulus_id':sess_dat.stimulus_id.values,
            #                  'stimulus_num':sess_dat.stimulus_num.values,
            #                  'sentence':sess_dat.sentence.values,
            #                  'stimulus':sess_dat.stimulus.values
            #                  }
            #     rsa_dataset.append(rsd.Dataset(measurements=sess_dat.values, descriptors=descriptors, obs_descriptors=obs_descriptors,
            #                                channel_descriptors=ch_descriptors))

        # for _,sess_dat in sub_dat.groupby('repeat_id'):
        #     True
        #     descriptors={'subject':sess_dat.subject.values[0],
        #              'session':sess_dat.repeat_id.values[0],
        #              'day':sess_dat.sess_id.values[0]
        #              }
        #     ch_descriptors={'neuroid':sess_dat.neuroid_id.values,
        #                 'roi':sess_dat.roi.values}
        #
        #     obs_descriptors={'stimulus_id':sess_dat.stimulus_id.values,
        #                  'stimulus_num':sess_dat.stimulus_num.values,
        #                  'sentence':sess_dat.sentence.values,
        #                  'stimulus':sess_dat.stimulus.values
        #                  }
        #     rsa_dataset.append(rsd.Dataset(measurements=sess_dat.values, descriptors=descriptors, obs_descriptors=obs_descriptors,
        #                                channel_descriptors=ch_descriptors))
        return rsa_dataset

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_", raw_kwargs=None):

        NotImplementedError
        # ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id, sha1=sha1)
        # if raw_kwargs:  # recursively load raw attributes
        #     raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
        #     ceiling.attrs['raw'] = raw
        # return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                          recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data[0].obs_descriptors
        stimuli_pd=pd.DataFrame(stimuli)
        stimuli=list(stimuli_pd.loc[stimuli_pd['session'] == 1].stimulus)
        stimul_id=np.asarray(stimuli_pd.loc[stimuli_pd['session'] == 1].stimulus_id)
        assert all(stimul_id[i] <= stimul_id[i + 1] for i in range(len(stimul_id) - 1))
        predictions=[]
        for idx,stim in tqdm(enumerate(stimuli),desc='digest individual sentences'):
            prediction = candidate.digest_text(str(stim))['neural']
            prediction['stimulus_id'] = 'presentation', np.repeat(stimul_id[idx],prediction.shape[0])
            predictions.append(prediction)
        predictions = xr.concat(predictions, dim='presentation')
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

        raw_score = self.metric(predictions_rsd, self.data)
        score = Score([raw_score.get_means(), raw_score.get_errorbars()[0]], coords={'aggregation': ['center', 'error'],},
                       dims=['aggregation','layer'])
        score.attrs['raw'] = raw_score
        score.attrs['noise_ceiling']=raw_score.get_noise_ceil()

        return score
