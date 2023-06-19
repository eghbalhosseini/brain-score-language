import logging

import numpy as np
from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from .package_activations import ACTIVATON_DIR
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from scipy.stats import median_abs_deviation
from tqdm import tqdm
import xarray as xr
from pathlib import Path
import pickle

BIBTEX = """
}"""

def DsParametric_fMRI(atlas,ceiling,stimulus_set):
    return _DsParametric_fMRI_ExperimentLinear(atlas=atlas,ceiling_s3_kwargs=ceiling,stimulus_set=stimulus_set)

class _DsParametric_fMRI_ExperimentLinear(BenchmarkBase):
    def __init__(self, atlas:str,stimulus_set:str,ceiling_s3_kwargs: dict ):
        self.stimulus_set = stimulus_set
        self.data = self._load_data(atlas)
        self.metric = load_metric('linear_pearsonr')

        identifier = f'DsParametric_fMRI.{atlas}-linear'
        ceiling = None #if not ceiling_s3_kwargs else self._load_ceiling(identifier=identifier, **ceiling_s3_kwargs)
        super(_DsParametric_fMRI_ExperimentLinear, self).__init__(
            identifier=identifier,
            version=1,
            parent='DsParametric_fMRI-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self, atlas: str) -> NeuroidAssembly:
        data = load_dataset(f'DsParametric_fMRI.{atlas}')
        data = data.dropna('neuroid')  # not all subjects have done both experiments, drop those that haven't
        data = data.sortby('stimulus_id')
        data = data[data['stim_group'] == self.stimulus_set]
        data.attrs['identifier'] = f'DsParametric_fMRI.{self.stimulus_set}.{atlas}'
        return data
    def __call__(self, candidate: ArtificialSubject,stim_set:str) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']

        # select data with stim_set
        predictions=[]
        # if the preditions are already computed, retrieve them
        prediction_path=Path(ACTIVATON_DIR,f'model={candidate.identifier}_stimuli=DsParametricfMRI_{self.stimulus_set}.pkl')
        if prediction_path.exists():
            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)

        else:
            for stim_id, stim in tqdm(stimuli.groupby('stimulus_id'), desc='digest individual sentences'):
                prediction = candidate.digest_text(str(stim.values))['neural']
                prediction['stimulus_id'] = 'presentation', stim['stimulus_id'].values
                predictions.append(prediction)
            predictions = xr.concat(predictions, dim='presentation')

        # get last word activations for predictions
        prediction_last_word=[]
        for idx, x in predictions.groupby('stim_name'):
            True
            assert(x['word'][-1]==str(x['stimulus_sentence'][-1].values).split()[-1])
            prediction_last_word.append(x[-1:,:])
        prediction_last_word=xr.concat(prediction_last_word,dim='presentation')
        # sort the prediction_last_word based on stim_name in self.data
        prediction_last_word=prediction_last_word.sortby('stimulus_id')
        assert(np.all(prediction_last_word['stim_name'].values==self.data['stim_name'].values))
        raw_scores = []
        for layer_id, prediction_lw in prediction_last_word.groupby('layer'):
            True
            raw_score = self.metric(prediction_lw, self.data)
            raw_scores.append(raw_score.raw.expand_dims(dim={"layer": [layer_id]}, axis=0))
        raw_scores = xr.concat(raw_scores, dim='layer')
        score = raw_scores.mean('split')
        score = score.groupby('subject').median()
        center = score.median('subject')
        subject_values = np.nan_to_num(score.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = score.dims.index(score['subject'].dims[0])
        error = median_abs_deviation(subject_values, axis=subject_axis)
        scores=[]
        if self.ceiling is None:
            scores = center
        else:
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

