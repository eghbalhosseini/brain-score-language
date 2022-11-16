from typing import  Tuple
import torch
import xarray as xr
from tqdm import tqdm
from collections import OrderedDict
from typing import Union, List, Tuple, Dict, Callable
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject, HuggingfaceGroup

def get_layer_names(model_id):
    model_ = AutoModelForMaskedLM.from_pretrained(model_id)
    flat_list = ('embeddings',) + tuple(
        f'encoder.albert_layer_groups.{i}' for i in range(model_.config.num_hidden_layers))
    return flat_list

activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class ALBERTSubject(HuggingfaceSubject):
    def __init__(self,model_id: str,
            region_layer_mapping: dict):
        super(ALBERTSubject, self).__init__(model_id,region_layer_mapping,model=AutoModelForMaskedLM.from_pretrained(model_id))
        self._set_albert_layers()

    def _set_albert_layers(self):
        # albert shares the same weight across layers so there is not apparent layer ordering and hooks dont work
        self.layers_names = ('embeddings',) + tuple(f'encoder.albert_layer_groups.{i}' for i in range(self.basemodel.config.num_hidden_layers))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        """
        if type(text) == str:
            text = [text]
        output = {'behavior': [], 'neural': []}
        number_of_tokens = 0
        text_iterator = tqdm(text, desc='digest text') if len(text) > 100 else text  # show progress bar if many parts
        for part_number, text_part in enumerate(text_iterator):
            # prepare string representation of context
            context = self._prepare_context(text[:part_number + 1])
            context_tokens, number_of_tokens = self._tokenize(context, number_of_tokens)
            layer_representations = OrderedDict()
            # run and remove hooks
            with torch.no_grad():
                base_output_and_hidden = self.basemodel(**context_tokens,output_hidden_states=True)

            base_output=base_output_and_hidden['logits']
            base_hidden=base_output_and_hidden['hidden_states']
            assert(len(base_hidden)==len(self.layers_names))
            layer_representations=self.get_hooks(base_hidden,layer_representations)
            # format output
            stimuli_coords = {
                'stimulus': ('presentation', [text_part]),
                'context': ('presentation', [context]),
                'part_number': ('presentation', [part_number]),
            }
            if self.behavioral_task:
                behavioral_output = self.output_to_behavior(base_output=base_output)
                behavior = BehavioralAssembly(
                    [behavioral_output],
                    coords=stimuli_coords,
                    dims=['presentation']
                )
                output['behavior'].append(behavior)
            if self.neural_recordings:
                representations = self.output_to_representations(layer_representations, stimuli_coords=stimuli_coords)
                output['neural'].append(representations)

        # merge over text parts
        self._logger.debug("Merging outputs")
        output['behavior'] = xr.concat(output['behavior'], dim='presentation').sortby('part_number') \
            if output['behavior'] else None
        output['neural'] = xr.concat(output['neural'], dim='presentation').sortby('part_number') \
            if output['neural'] else None
        return output

        #TODO see https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/ for assigninig layers

    def get_hooks(self,states,layer_representations):
        for (recording_target, recording_type) in self.neural_recordings:
            layer_names = self.region_layer_mapping[recording_target]
            if type(layer_names) == str:
                layer_names = [layer_names]
            for layer_name in layer_names:
                layer_loc=self.layers_names.index(layer_name)
                layer_representations[(recording_target,recording_type,layer_name)]=states[layer_loc]
        return layer_representations




class ALBERTGroup(ALBERTSubject):
    def __init__(self, model_id: str,
                 region_layer_mapping: dict):
        super(ALBERTGroup, self).__init__(model_id, region_layer_mapping)

    def get_hooks(self,states,layer_representations):
        for (recording_target, recording_type) in self.neural_recordings:
            layer_names = self.region_layer_mapping[recording_target]
            if type(layer_names) == str:
                layer_names = [layer_names]
            for layer_name in layer_names:
                layer_loc=self.layers_names.index(layer_name)
                layer_representations[(recording_target,recording_type,layer_name)]=states[layer_loc]
        return layer_representations