import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from brainio.assemblies import NeuroidAssembly, array_is_element, DataAssembly
from brainio.assemblies import walk_coords
from brainscore_core.metrics import Score, Metric
from brainscore_language.utils.transformations import CrossValidation


class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'stimulus_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


class XarrayRSA:
    def __init__(self, rsa, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._rsa = rsa
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None