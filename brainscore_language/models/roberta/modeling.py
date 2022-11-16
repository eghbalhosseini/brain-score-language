from typing import  Tuple
import torch
from collections import OrderedDict
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject, HuggingfaceGroup
import re

def get_layer_names(model_id):
    model_ = AutoModelForMaskedLM.from_pretrained(model_id)
    modul_names = [x[0] for x in model_.named_modules()]
    layer_drop_names = [x for x in modul_names if len(re.findall(r'roberta.embeddings.dropout$', x)) > 0]
    layer_output_names = [x for x in modul_names if len(re.findall(r'roberta.encoder.layer.\d+$', x)) > 0]
    flat_list = [item for sublist in [layer_drop_names,layer_output_names] for item in sublist]
    return flat_list

class RoBERTaSubject(HuggingfaceSubject):
    def __init__(self,model_id: str,
            region_layer_mapping: dict):
        super(RoBERTaSubject, self).__init__(model_id,region_layer_mapping,model=AutoModelForMaskedLM.from_pretrained(model_id))

    def _register_hook(self,
                       layer: torch.nn.Module,
                       key: Tuple[str, str, str],
                       target_dict: dict) -> RemovableHandle:
        # instantiate parameters to function defaults; otherwise they would change on next function call
        def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
            if len(output)>1:
                # fix for when taking out only the hidden state, thisis different from droput because of residual state
                # see :  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
                target_dict[key] = self._tensor_to_numpy(output[0])
            elif type(output)==tuple:
                target_dict[key] = self._tensor_to_numpy(output[0])
            else:
                target_dict[key] = self._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

class RoBERTaGroup(RoBERTaSubject):
    def __init__(self, model_id: str,
                 region_layer_mapping: dict):
        super(RoBERTaGroup, self).__init__(model_id, region_layer_mapping)

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