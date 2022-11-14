from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.albert.modeling import ALBERTSubject,ALBERTGroup
from transformers import AutoModelForMaskedLM
import re

def get_layer_names(model_id):
    model_ = AutoModelForMaskedLM.from_pretrained(model_id)
    flat_list = ('embeddings',) + tuple(
        f'encoder.albert_layer_groups.{i}' for i in range(model_.config.num_hidden_layers))
    return flat_list

model_id="albert-xxlarge-v2"
model_registry[model_id] = lambda: ALBERTSubject(model_id=model_id, region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'encoder.albert_layer_groups.3'})


model_id="albert-xxlarge-v2"
model_registry[f'{model_id}-layerwise'] = lambda: ALBERTGroup(model_id=model_id, region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names(model_id)})