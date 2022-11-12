from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject,HuggingfaceGroup
from transformers import AutoModelForCausalLM
import re


def get_layer_names(model_id):
    model_ = AutoModelForCausalLM.from_pretrained(model_id)
    modul_names = [x[0] for x in model_.named_modules()]
    layer_drop_names = [x for x in modul_names if len(re.findall(r'.drop$', x)) > 0]
    layer_output_names = [x for x in modul_names if len(re.findall(r'h.\d+$', x)) > 0]
    flat_list = [item for sublist in [layer_drop_names,layer_output_names] for item in sublist]
    return flat_list

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry['distilgpt2'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5.mlp.dropout'})

model_registry['distilgpt2-Pereira2018'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'})

model_id='distilgpt2'
model_registry[f'{model_id}-layerwise'] = lambda: HuggingfaceGroup(model_id=model_id, region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names(model_id)})


model_registry['gpt2-xl'] = lambda: HuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43.mlp.dropout'})
