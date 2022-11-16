import logging
import sys
import warnings
from collections import defaultdict

import numpy as np
from numpy import AxisError
from numpy.random import RandomState
from result_caching import store
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
import xarray as xr
from brainio.assemblies import DataAssembly, array_is_element, walk_coords, merge_data_arrays
from brainscore_core.metrics import Score
from brainscore_language import load_benchmark
from brainscore_language.utils import fullname
from brainscore_language.utils.s3 import upload_data_assembly
from brainscore_language.utils.transformations import apply_aggregate
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import pandas as pd

_logger = logging.getLogger(__name__)

def upload_ceiling(atlas):
    benchmark = load_benchmark(f'ANNSet1_fMRI.{atlas}-linear')
    ceiler = ExtrapolationCeiling()
    ceiling = ceiler(benchmark.data, metric=benchmark.metric)
    _logger.info(f"Uploading ceiling {ceiling} and saving them locally")
    ceiling_dir = Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ceiling_{benchmark.identifier}.pkl')
    with open(ceiling_dir.__str__(), 'wb') as f:
        pickle.dump(ceiling,f)
    # Note that because we cannot serialize complex objects to netCDF, attributes like 'raw' and 'bootstrapped_params'
    # will get lost during upload
    upload_data_assembly(ceiling,
                         assembly_identifier=benchmark.identifier,
                         assembly_prefix='ceiling_')
    # also upload raw (and raw.raw) attributes
    upload_data_assembly(ceiling.raw,
                         assembly_identifier=benchmark.identifier,
                         assembly_prefix='ceiling_raw_')

    #upload_data_assembly(ceiling.raw.raw,
    #                     assembly_identifier=benchmark.identifier,
    #                     assembly_prefix='ceiling_raw_raw_')


def v(x, v0, tau0):  # function for ceiling extrapolation
    return v0 * (1 - np.exp(-x / tau0))


class HoldoutSubjectCeiling:
    def __init__(self, subject_column):
        self.subject_column = subject_column
        self._logger = logging.getLogger(fullname(self))
        self._rng = RandomState(0)
        self._num_bootstraps = 5

    def __call__(self, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        scores = []
        '''give a set of subjects create a random selection, say we have {906,916} the selection would be ['916', '916', '916', '916', '906'] '''
        iterate_subjects = self._rng.choice(list(subjects), size=self._num_bootstraps)  # use only a subset of subjects
        for subject in tqdm(iterate_subjects, desc='heldout subject'):
            try:
                '''procedure here is as follow: 
                1. create a subject_assembly which is the subject to be predicted, 906
                2. create a pool assembly which is the pool of subjects to use to predict subject_assembly its subjects - subject
                3. compute a metric between subject_assembly and pool assembly
                4. setting the subject coordinate
                '''
                subject_assembly = assembly[{'neuroid': [subject_value == subject
                                                         for subject_value in assembly[self.subject_column].values]}]
                # run subject pool as neural candidate
                subject_pool = subjects - {subject}
                pool_assembly = assembly[
                    {'neuroid': [subject in subject_pool for subject in assembly[self.subject_column].values]}]
                score = metric(pool_assembly, subject_assembly)
                # store scores
                apply_raw = 'raw' in score.attrs and \
                            not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
                scores.append(score)
            except NoOverlapException as e:
                self._logger.debug(f"Ignoring no overlap {e}")
                continue  # ignore
            except ValueError as e:
                if "Found array with" in str(e):
                    self._logger.debug(f"Ignoring empty array {e}")
                    continue
                else:
                    raise e
        '''combine voxel scores hold out subjects : ['916', '916', '916', '916', '906'] , given the reptitition it would be sum of 906 and 916'''
        scores = Score.merge(*scores)
        scores = apply_aggregate(lambda scores: scores.mean(self.subject_column), scores)
        return scores


class ExtrapolationCeiling:
    def __init__(self, subject_column='subject', extrapolation_dimension='neuroid',
                 num_bootstraps=100):
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.extrapolation_dimension = extrapolation_dimension
        self.num_bootstraps = num_bootstraps
        self.num_subsamples = 10
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self._rng = RandomState(0)

    def __call__(self, assembly, metric):
        scores = self.collect(identifier=assembly.identifier, assembly=assembly, metric=metric)
        return self.extrapolate(identifier=assembly.identifier, ceilings=scores)

    @store(identifier_ignore=['assembly', 'metric'])
    def collect(self, identifier, assembly, metric):
        self._logger.debug("Collecting data for extrapolation")
        subjects = set(assembly[self.subject_column].values)
        subject_subsamples = tuple(range(2, len(subjects) + 1))
        scores = []
        scores_aggregate=[]
        '''iterate over the subject sets, say we have 8 subjects then we can create pools with (2, 3, 4, 5, 6, 7, 8) subjects in them'''
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            ''' create a pool of subjects based on the number of subjects given, example (2 subjects : {('837', '865'),
            ('837', '906'),('837', '913'),('848', '865'),('906', '682'),('906', '848'),('913', '865'),('916', '848'),
            ('916', '880'),('916', '906')}
            '''
            subject_sets =  self._random_combinations(subjects=set(assembly[self.subject_column].values),num_subjects=num_subjects, choice=self.num_subsamples, rng=self._rng)
            for sub_subjects in subject_sets:
                sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                     for subject in assembly[self.subject_column].values]}]
                selections = {self.subject_column: sub_subjects}
                try:
                    score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
                    score = score.expand_dims('num_subjects')
                    score['num_subjects'] = [num_subjects]
                    for key, selection in selections.items():
                        expand_dim = f'sub_{key}'
                        score = score.expand_dims(expand_dim)
                        score[expand_dim] = [str(selection)]
                    scores.append(score.raw)
                    scores_aggregate.append(score)
                except KeyError as e:  # nothing to merge
                    if str(e) == "'z'":
                        self._logger.debug(f"Ignoring merge error {e}")
                        continue
                    else:
                        raise e

        scores_xr = Score.merge(*scores)
        scores_xr.attrs['raw_score'] = scores
        scores_xr.attrs['raw_aggregate_score'] = scores_aggregate
        '''this section was used to print the results of estimates
        BEGIN CODE 
        x_val=np.stack([x.num_subjects.values for x in scores_aggregate]).squeeze()
        y_val=np.stack([x.values for x in scores_aggregate]).squeeze()
        fig = plt.figure(figsize=[11, 8])
        ax0 = fig.add_axes((.2, .1, .6, .7))
        ax0.scatter(x_val+np.random.normal(0,.02,x_val.shape),y_val,s=50,c='r',edgecolor='k')

        ax0.set_xlabel("number of subject")
        ax0.set_ylabel("var explained, left out subject")
        ax0.axhline(y=0, color='k', linewidth=2)
        ax0.set_ylim([-.1,.6])

        fig.savefig(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{assembly.identifier}_regression_ceilings.png',
                    facecolor=(1, 1, 1), edgecolor='none')
        fig.savefig(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/plots/{assembly.identifier}_regression_ceilings.pdf',
                   facecolor=(1, 1, 1), edgecolor='none')
                   
        END CODE 
        '''
        return scores_xr

    def _random_combinations(self, subjects, num_subjects, choice, rng):
        # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
        subjects = np.array(list(subjects))
        combinations = set()
        while len(combinations) < choice:
            elements = rng.choice(subjects, size=num_subjects, replace=False)
            combinations.add(tuple(elements))
        return combinations

    #@store(identifier_ignore=['assembly', 'metric'])
    def extrapolate(self, identifier, ceilings):
        neuroid_ceilings = []
        raw_keys = ['bootstrapped_params', 'error_low', 'error_high', 'endpoint_x']
        raw_attrs = defaultdict(list)
        for i in trange(len(ceilings[self.extrapolation_dimension]),
                        desc=f'{self.extrapolation_dimension} extrapolations'):
            try:
                # extrapolate per-neuroid ceiling
                neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
                extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling.squeeze())
                extrapolated_ceiling = self.add_neuroid_meta(extrapolated_ceiling, neuroid_ceiling)
                neuroid_ceilings.append(extrapolated_ceiling)
                # keep track of raw attributes
                for key in raw_keys:
                    values = extrapolated_ceiling.attrs[key]
                    values = self.add_neuroid_meta(values, neuroid_ceiling)
                    raw_attrs[key].append(values)
            except AxisError:  # no extrapolation successful (happens for 1 neuroid in Pereira)
                _logger.warning(f"Failed to extrapolate neuroid ceiling {i}", exc_info=True)
                continue

        # merge and add meta
        self._logger.debug("Merging neuroid ceilings")
        neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['raw'] = ceilings

        for key, values in raw_attrs.items():
            self._logger.debug(f"Merging {key}")
            values = manual_merge(*values, on=self.extrapolation_dimension)
            neuroid_ceilings.attrs[key] = values
        # aggregate
        ceiling = self.aggregate_neuroid_ceilings(neuroid_ceilings, raw_keys=raw_keys)
        ceiling.attrs['identifier'] = identifier
        return ceiling

    def add_neuroid_meta(self, target, source):
        target = target.expand_dims(self.extrapolation_dimension)
        for coord, dims, values in walk_coords(source):
            if array_is_element(dims, self.extrapolation_dimension):
                target[coord] = dims, values
        return target

    def aggregate_neuroid_ceilings(self, neuroid_ceilings, raw_keys):
        ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
        ceiling.attrs['raw'] = neuroid_ceilings
        for key in raw_keys:
            values = neuroid_ceilings.attrs[key]
            aggregate = values.median(self.extrapolation_dimension)
            if not aggregate.shape:  # scalar value, e.g. for error_low
                aggregate = aggregate.item()
            ceiling.attrs[key] = aggregate
        return ceiling

    def extrapolate_neuroid(self, ceilings):
        # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        rng = RandomState(0)
        bootstrap_params = []
        for bootstrap in range(self.num_bootstraps):
            bootstrapped_scores = []
            for num_subjects in subject_subsamples:
                num_scores = ceilings.sel(num_subjects=num_subjects)
                # the sub_subjects dimension creates nans, get rid of those
                num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'} or \
                       set(num_scores.dims) == {f'sub_{self.subject_column}'}
                # choose from subject subsets and the splits therein, with replacement for variance
                choices = num_scores.values.flatten()
                bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                bootstrapped_scores.append(np.mean(bootstrapped_score))

            try:
                params = self.fit(subject_subsamples, bootstrapped_scores)
            except :  # optimal parameters not found
                params = [np.nan, np.nan]
            params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                  dims=['bootstrap', 'param'])
            bootstrap_params.append(params)
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        if not np.isnan(bootstrap_params.values).all():
            ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                       if not np.isnan(params).any()])
            median_ys = np.median(ys, axis=0)
            diffs = np.diff(median_ys)
            end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
            center = np.median(np.array(bootstrap_params)[:, 0])
            error_low, error_high = ci_error(ys[:, end_x], center=center)
            score = Score(center)
            score.attrs['raw'] = ceilings
            score.attrs['error_low'] = DataAssembly(error_low)
            score.attrs['error_high'] = DataAssembly(error_high)
            score.attrs['bootstrapped_params'] = bootstrap_params
            score.attrs['endpoint_x'] = DataAssembly(end_x)
        else:
            score = Score(np.asarray(np.nan))
            score.attrs['raw'] = ceilings
            score.attrs['error_low'] = DataAssembly(np.asarray(np.nan))
            score.attrs['error_high'] = DataAssembly(np.asarray(np.nan))
            score.attrs['bootstrapped_params'] = bootstrap_params
            score.attrs['endpoint_x'] = DataAssembly(np.asarray(np.nan))
        return score

    def fit(self, subject_subsamples, bootstrapped_scores):
        valid = ~np.isnan(bootstrapped_scores)
        if sum(valid) < 1:
            raise RuntimeError("No valid scores in sample")
        params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                 bounds=([0, -np.inf], [1, np.inf]))
        return params


def manual_merge(*elements, on='neuroid'):
    dims = elements[0].dims
    assert all(element.dims == dims for element in elements[1:])
    merge_index = dims.index(on)
    # the coordinates in the merge index should have the same keys
    assert _coords_match(elements, dim=on,
                         match_values=False), f"coords in {[element[on] for element in elements]} do not match"
    # all other dimensions, their coordinates and values should already align
    for dim in set(dims) - {on}:
        assert _coords_match(elements, dim=dim,
                             match_values=True), f"coords in {[element[dim] for element in elements]} do not match"
    # merge values without meta
    merged_values = np.concatenate([element.values for element in elements], axis=merge_index)
    # piece together with meta
    result = type(elements[0])(merged_values, coords={
        **{coord: (dims, values)
           for coord, dims, values in walk_coords(elements[0])
           if not array_is_element(dims, on)},
        **{coord: (dims, np.concatenate([element[coord].values for element in elements]))
           for coord, dims, _ in walk_coords(elements[0])
           if array_is_element(dims, on)}}, dims=elements[0].dims)
    return result


def _coords_match(elements, dim, match_values=False):
    first_coords = [(key, tuple(value)) if match_values else key for _, key, value in walk_coords(elements[0][dim])]
    other_coords = [[(key, tuple(value)) if match_values else key for _, key, value in walk_coords(element[dim])]
                    for element in elements[1:]]
    return all(tuple(first_coords) == tuple(coords) for coords in other_coords)


def ci_error(samples, center, confidence=.95):
    low, high = 100 * ((1 - confidence) / 2), 100 * (1 - ((1 - confidence) / 2))
    confidence_below, confidence_above = np.nanpercentile(samples, low), np.nanpercentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


class NoOverlapException(Exception):
    pass

parser = argparse.ArgumentParser(description='ceiling packaging for the benchmark')
parser.add_argument('atlas', type=str)
args=parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['botocore', 'boto3', 's3transfer', 'urllib3']:
        logging.getLogger(shush_logger).setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    atlas=str(args.atlas)
    upload_ceiling(atlas)

    #upload_ceiling('train.language_top_70')
    #upload_ceiling('train.language_top_70')

    # benchmark = load_benchmark(f'ANNSet1_fMRI.train.language_top_90-linear')
    # ceiler = ExtrapolationCeiling()
    # ceiling = ceiler(benchmark.data, metric=benchmark.metric)
    # scores=ceiler.collect(identifier=benchmark.data.identifier,assembly=benchmark.data,metric=benchmark.metric)
    # ceilings=ceiler.extrapolate(identifier=benchmark.data.identifier, ceilings=scores)

    # ceiling_dir=Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ceiling_{benchmark.identifier}.pkl')
    # with open(ceiling_dir.__str__(),'wb') as f:
    #    pickle.dump(ceilings,f)
    #     _logger.info(f"Uploading ceiling {ceiling}")
    #     # Note that because we cannot serialize complex objects to netCDF, attributes like 'raw' and 'bootstrapped_params'
    #     # will get lost during upload
    # upload_data_assembly(ceilings,
    #                     assembly_identifier=benchmark.identifier,
    #                      assembly_prefix='ceiling_')
    #     # also upload raw (and raw.raw) attributes
    # upload_data_assembly(ceilings.raw,
    #                      assembly_identifier=benchmark.identifier,
    #                      assembly_prefix='ceiling_raw_')
    # upload_data_assembly(ceilings.raw.raw,
    #                      assembly_identifier=benchmark.identifier,
    #                      assembly_prefix='ceiling_raw_raw_')