import logging

import numpy as np
import xarray as xr
from brainio.assemblies import NeuroidAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric, benchmark_registry
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.utils.s3 import load_from_s3
from scipy.stats import median_abs_deviation
from tqdm import tqdm
logger = logging.getLogger(__name__)

BIBTEX = """@article{Fedorenko2016-od,
  title={Neural correlate of the construction of sentence meaning},
  author={Fedorenko, Evelina and Scott, Terri L and Brunner, Peter and
              Coon, William G and Pritchett, Brianna and Schalk, Gerwin and
              Kanwisher, Nancy},
  journal={Proc. Natl. Acad. Sci. U. S. A.},
  volume={113},
  number={41},
  pages={E6256--E6262},
  month={oct},
  year={2016}
}"""

def ceiling_electrodes(raw_score: Score, ceiling: Score) -> Score:
    # normalize by ceiling, but not above 1
    score = raw_score / ceiling
    score.attrs['raw'] = raw_score
    score.attrs['ceiling'] = ceiling
    if score > 1:
        overshoot_value = score.item()
        # ideally we would just update the value, but I could not figure out how to update a scalar DataArray
        attrs = score.attrs
        score = type(score)(1, coords=score.coords, dims=score.dims)
        score.attrs = attrs
        score.attrs['overshoot'] = overshoot_value
    return score

def Fedorenko2016_ECoG():
    return _Fedorenko2016ExperimentLinear( ceiling_s3_kwargs=dict(
        version_id='lBAD_QPrwqix0BeGzf9BC41gtzC2VhOY',
        sha1='64509b5065eea23f6f904c50f64ae9c66ebd0c7f',
        raw_kwargs=dict(
            version_id='f4jR1axxH3YXGCur8E3KJeWkU364uxLY',
            sha1='ed1c6cc24da2210ede263e528678d15e3f70deb8',
            raw_kwargs=dict(
                version_id='D6pT1.5XDuNlFekdsuu68CXr5y9Z.M.6',
                sha1='1f531641566ec1d6120d47b22cb421b4e7b70f43')
        )
    ))

class _Fedorenko2016ExperimentLinear(BenchmarkBase):

    def __init__(self, ceiling_s3_kwargs: dict):
        self.data=self._load_data()
        self.metric=load_metric('linear_pearsonr')
        self.metric.cross_validation._split._split.n_splits=5
        identifier=f'Fedorenko2016-linear'
        ceiling=self._load_ceiling(identifier=identifier,**ceiling_s3_kwargs)
        super(_Fedorenko2016ExperimentLinear,self).__init__(
            identifier=identifier,
            version=1,
            parent='Fedorenko2016-linear',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def _load_data(self) ->  NeuroidAssembly:
        data=load_dataset('Fedorenko2016.language')
        data=data.dropna('neuroid')
        data.attrs['identifier']= f'{data.identifier}'
        return data

    def _load_ceiling(self, identifier: str, version_id: str, sha1: str, assembly_prefix="ceiling_",
                      raw_kwargs=None):
        ceiling = load_from_s3(identifier, cls=Score, assembly_prefix=assembly_prefix, version_id=version_id,
                               sha1=sha1)
        if raw_kwargs:  # recursively load raw attributes
            raw = self._load_ceiling(identifier=identifier, assembly_prefix=assembly_prefix + "raw_", **raw_kwargs)
            ceiling.attrs['raw'] = raw
        return ceiling

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                           recording_type=ArtificialSubject.RecordingType.ECoG)
        stimuli = self.data['stimulus']
        sentences = self.data['sentence_id'].values
        predictions = []

        for sentence in tqdm(sorted(set(sentences)), desc='digest individual sentences'):
            sentence_indexer = [sentence_id == sentence for sentence_id in sentences]
            sentence_stimuli = stimuli[sentence_indexer]
            assert np.array_equal(sentence_stimuli.word_num.values, np.sort(sentence_stimuli.word_num.values))
            sentence_prediction = candidate.digest_text(sentence_stimuli.values)['neural']
            sentence_prediction['stimulus_id'] = 'presentation', sentence_stimuli['stimulus_id'].values
            predictions.append(sentence_prediction)
        predictions = xr.concat(predictions, dim='presentation')
        raw_scores=[]
        for layer_id, prediction in predictions.groupby('layer'):
            raw_score = self.metric(prediction, self.data)
            raw_scores.append(raw_score.raw.expand_dims(dim={"layer":[layer_id]},axis=0))
        raw_scores=xr.concat(raw_scores,dim='layer')
        score_by_electrode=raw_scores.mean('split')
        ceiling_by_electrode=self.ceiling.raw.sel(aggregation='center')
        score = score_by_electrode / ceiling_by_electrode
        score_by_subject=score.groupby('subject_UID').median()
        center=score_by_subject.median('subject_UID')
        subject_values = np.nan_to_num(score_by_subject.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = score_by_subject.dims.index(score_by_subject['subject_UID'].dims[0])
        error = median_abs_deviation(subject_values, axis=subject_axis)
        score = Score([center, error], coords={'aggregation': ['center', 'error'],
                                               'layer':raw_scores.layer.values},
                      dims=['aggregation','layer'])
        score.attrs['raw']=raw_scores
        score.attrs['ceiling']=self.ceiling.raw
        score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                     "then median of subject scores"
        return score

benchmark_registry['Fedorenko2016-linear'] = Fedorenko2016_ECoG