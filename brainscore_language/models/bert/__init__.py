from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from .modeling import BERTSubject, BERTGroup, get_layer_names

model_registry['bert-large-uncased-whole-word-masking'] = lambda: BERTSubject(model_id="bert-large-uncased-whole-word-masking", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.11.output'})


model_registry['bert-base-uncased'] = lambda: BERTSubject(model_id="bert-base-uncased", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.11.output'})

model_registry['bert-base-uncased-layerwise'] = lambda: BERTGroup(model_id='bert-base-uncased', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names('bert-base-uncased')})