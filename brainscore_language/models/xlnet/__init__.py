from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.xlnet.modeling import XLNetSubject
from transformers import AutoModelForCausalLM
import re

def get_layer_names(model_id):
    model_ = AutoModelForCausalLM.from_pretrained(model_id)
    modul_names = [x[0] for x in model_.named_modules()]
    layer_drop_names = [x for x in modul_names if len(re.findall(r'.word_embedding$', x)) > 0]
    layer_output_names = [x for x in modul_names if len(re.findall(r'transformer.layer.\d+$', x)) > 0]
    flat_list = [item for sublist in [layer_drop_names,layer_output_names] for item in sublist]
    return flat_list


model_id='xlnet-large-cased'
get_layer_names(model_id)

model_registry[model_id] = lambda: XLNetSubject(model_id=model_id, region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.layer.23'})