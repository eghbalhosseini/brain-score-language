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

def ANNSet1_fMRI_benchmarkLinear(atlas=None):
    return ANNSet1_fMRI_ExperimentLinear(atlas)


class ANNSet1_fMRI_ExperimentLinear(BenchmarkBase):
    def __init__(self, atlas:str ):
        self.data = self._load_data(atlas)
        self.metric = load_metric('linear_pearsonr')
        identifier = f'ANNSet1_fMRI.{atlas}-linear'
        ceiling = None#self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(ANNSet1_fMRI_ExperimentLinear, self).__init__(
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
        score_by_voxels = raw_scores.mean('split')
        score_by_subject = score_by_voxels.groupby('subject').median()
        center = score_by_subject.median('subject')
        subject_values = np.nan_to_num(score_by_subject.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = score_by_subject.dims.index(score_by_subject['subject'].dims[0])
        error = median_abs_deviation(subject_values, axis=subject_axis)
        score = Score([center, error], coords={'aggregation': ['center', 'error'],
                                               'layer': raw_scores.layer.values},
                      dims=['aggregation', 'layer'])
        score.attrs['raw'] = raw_scores
        score.attrs['ceiling'] = self.ceiling
        score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                     "then median of subject scores"

        return score
