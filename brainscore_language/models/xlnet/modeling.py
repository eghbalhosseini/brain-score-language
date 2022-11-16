import functools
from typing import Union, List, Tuple, Dict, Callable
import numpy as np
from numpy.core import defchararray
from brainio.assemblies import  NeuroidAssembly
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from collections import OrderedDict
from transformers import AutoModelForCausalLM
import re

def get_layer_names(model_id):
    model_ = AutoModelForCausalLM.from_pretrained(model_id)
    modul_names = [x[0] for x in model_.named_modules()]
    layer_drop_names = [x for x in modul_names if len(re.findall(r'.word_embedding$', x)) > 0]
    layer_output_names = [x for x in modul_names if len(re.findall(r'transformer.layer.\d+$', x)) > 0]
    flat_list = [item for sublist in [layer_drop_names,layer_output_names] for item in sublist]
    return flat_list

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


class XLNetGroup(XLNetSubject):
    def __init__(self, model_id: str,
                 region_layer_mapping: dict):
        super(XLNetGroup, self).__init__(model_id, region_layer_mapping)

    def _setup_hooks(self):
        """ set up a group of hooks for recording internal neural activity from the model (aka layer activations) """
        hooks = []
        layer_representations = OrderedDict()
        for (recording_target, recording_type) in self.neural_recordings:
            layer_names = self.region_layer_mapping[recording_target]
            if type(layer_names) == str:
                layer_names = [layer_names]
            for layer_name in layer_names:
                layer = self._get_layer(layer_name)
                hook = self._register_hook(layer, key=(recording_target, recording_type, layer_name),
                                           target_dict=layer_representations)
                hooks.append(hook)
        return hooks, layer_representations
