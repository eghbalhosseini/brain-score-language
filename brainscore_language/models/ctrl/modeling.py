import numpy as np
import torch
from collections import OrderedDict
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

class CTRLSubject(HuggingfaceSubject):
    def __init__(self,model_id: str,
            region_layer_mapping: dict):
        super(CTRLSubject, self).__init__(model_id,region_layer_mapping)

    def _tokenize(self, context, num_previous_context_tokens):
        """
        Tokenizes the context, keeping track of the newly added tokens in `self.current_tokens`
        """
        context_tokens = self.tokenizer(context, truncation=True, return_tensors="pt")
        context_tokens.to('cuda' if torch.cuda.is_available() else 'cpu')
        # keep track of tokens in current `text_part`
        overflowing_encoding: list = None if not np.array(context_tokens.encodings).item() else np.array(context_tokens.encodings).item().overflowing
        num_overflowing = 0 if not overflowing_encoding else sum(len(overflow) for overflow in overflowing_encoding)
        self.current_tokens = {key: value[..., num_previous_context_tokens - num_overflowing:]
                               for key, value in context_tokens.items()}
        num_new_context_tokens = context_tokens['input_ids'].shape[-1] + num_overflowing
        return context_tokens, num_new_context_tokens


class CTRLGroup(CTRLSubject):
    def __init__(self, model_id: str,
                 region_layer_mapping: dict):
        super(CTRLGroup, self).__init__(model_id, region_layer_mapping)

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