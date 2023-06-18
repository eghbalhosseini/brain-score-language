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

class _DsParametric_fMRI_ExperimentLinear(BenchmarkBase):
    def __init__(self, atlas:str,stimulus_set:str,ceiling_s3_kwargs: dict ):
        self.data = self._load_data(atlas)
        self.metric = load_metric('linear_pearsonr')
        self.stimulus_set=stimulus_set
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
        prediction_path=Path(ACTIVATON_DIR,f'model={candidate.identifier}_stimuli=DsParametricfMRI_{stim_set}.pkl')
        if prediction_path.exists():
            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)
        else:
            for stim_id, stim in tqdm(stimuli.groupby('stimulus_id'), desc='digest individual sentences'):
                prediction = candidate.digest_text(str(stim.values))['neural']
                prediction['stimulus_id'] = 'presentation', stim['stimulus_id'].values
                predictions.append(prediction)
            predictions = xr.concat(predictions, dim='presentation')

