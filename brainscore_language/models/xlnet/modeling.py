import functools
from typing import Union, List, Tuple, Dict, Callable
import numpy as np
from numpy.core import defchararray
from brainio.assemblies import  NeuroidAssembly
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject


class XLNetSubject(HuggingfaceSubject):
    def __init__(self,model_id: str,
            region_layer_mapping: dict):
        super(XLNetSubject, self).__init__(model_id,region_layer_mapping)

    def output_to_representations(self, layer_representations: Dict[Tuple[str, str, str], np.ndarray], stimuli_coords):
        representation_values = np.concatenate([
            # Choose to use last token (-1) of values[tokens,batch, token, unit] to represent passage.
            values[-1:, :, :].squeeze(0) for values in layer_representations.values()],
            axis=-1)  # concatenate along neuron axis
        neuroid_coords = {
            'layer': ('neuroid', np.concatenate([[layer] * values.shape[-1]
                                                 for (recording_target, recording_type, layer), values
                                                 in layer_representations.items()])),
            'region': ('neuroid', np.concatenate([[recording_target] * values.shape[-1]
                                                  for (recording_target, recording_type, layer), values
                                                  in layer_representations.items()])),
            'recording_type': ('neuroid', np.concatenate([[recording_type] * values.shape[-1]
                                                          for (recording_target, recording_type, layer), values
                                                          in layer_representations.items()])),
            'neuron_number_in_layer': ('neuroid', np.concatenate(
                [np.arange(values.shape[-1]) for values in layer_representations.values()])),
        }
        neuroid_coords['neuroid_id'] = 'neuroid', functools.reduce(defchararray.add, [
            neuroid_coords['layer'][1], '--', neuroid_coords['neuron_number_in_layer'][1].astype(str)])
        representations = NeuroidAssembly(
            representation_values,
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'])
        return representations