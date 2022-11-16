import logging

import numpy as np
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

logger = logging.getLogger(__name__)

BIBTEX = """
}"""
def ANNSet1_fMRI_lang_top_90():
    return _ANNSet1_fMRI_ExperimentLinear(atlas='train.language_top_90',ceiling_s3_kwargs=dict(
        version_id='L49MDfmlJCF7q5TvWI1S0n_NAvfFo5Zg',
        sha1='4c499cfa5491d75d93fc29d1e18ff12771b2bdbf',
        raw_kwargs=dict(version_id='eErH0hqDvGrUo5o79L1b4eqECXDzSlub',
            sha1='31f6035ae2d7f3734292ff4d35fccf7e92bd19ce')))

def ANNSet1_fMRI_lang_top_80():
    return _ANNSet1_fMRI_ExperimentLinear(atlas='train.language_top_80',ceiling_s3_kwargs=None)

def ANNSet1_fMRI_lang_top_70():
    return _ANNSet1_fMRI_ExperimentLinear(atlas='train.language_top_70',ceiling_s3_kwargs=None)

def ANNSet1_fMRI_auditory():
    return _ANNSet1_fMRI_ExperimentLinear(atlas='train.auditory',ceiling_s3_kwargs=None)

def ANNSet1_fMRI_visual():
    return _ANNSet1_fMRI_ExperimentLinear(atlas='train.visual',ceiling_s3_kwargs=None)


#def ANNSet1_fMRI_benchmarkLinear(atlas=None,ceiling_s3_kwargs=None):
#    return _ANNSet1_fMRI_ExperimentLinear(atlas,ceiling_s3_kwargs)


class _ANNSet1_fMRI_ExperimentLinear(BenchmarkBase):
    def __init__(self, atlas:str,ceiling_s3_kwargs: dict ):
        self.data = self._load_data(atlas)
        self.metric = load_metric('linear_pearsonr')
        identifier = f'ANNSet1_fMRI.{atlas}-linear'
        ceiling = None if not ceiling_s3_kwargs else self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_ANNSet1_fMRI_ExperimentLinear, self).__init__(
            identifier=identifier,
            version=1,
            parent='ANNSet1_fMRI-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, atlas: str) -> NeuroidAssembly:
        data = load_dataset(f'ANNSet1_fMRI.{atlas}')
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data = data.sortby('stimulus_id')
        data.attrs['identifier'] = f'ANNSet1_fMRI.{atlas}'
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
        predictions=[]
        for stim_id,stim in tqdm(stimuli.groupby('stimulus_id'),desc='digest individual sentences'):

            prediction = candidate.digest_text(str(stim.values))['neural']
            prediction['stimulus_id'] = 'presentation', stim['stimulus_id'].values
            predictions.append(prediction)
        predictions = xr.concat(predictions, dim='presentation')
        assert np.array_equal(predictions.stimulus_id.values, self.data.stimulus_id.values)

        raw_scores=[]
        for layer_id, prediction in predictions.groupby('layer'):
            raw_score = self.metric(prediction, self.data)
            raw_scores.append(raw_score.raw.expand_dims(dim={"layer":[layer_id]},axis=0))
        raw_scores = xr.concat(raw_scores, dim='layer')
        score = raw_scores.mean('split')
        score = score.groupby('subject').median()
        center = score.median('subject')
        subject_values = np.nan_to_num(score.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = score.dims.index(score['subject'].dims[0])
        error = median_abs_deviation(subject_values, axis=subject_axis)
        # ceiling normalize
        scores=[]
        for l,sc in center.groupby('layer'):
            sc = ceiling_normalize(sc, self.ceiling)
            scores.append(sc)
        scores=xr.concat(scores,dim='layer')
        score = Score([scores.values, error], coords={'aggregation': ['center', 'error'],
                                               'layer':scores.layer.values},
                      dims=['aggregation','layer'])
        score.attrs['raw'] = raw_scores
        score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                     "then median of subject scores"

        return score
