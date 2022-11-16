from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from .modeling import BERTSubject, BERTGroup, get_layer_names

model_registry["bert-base-uncased"] = lambda: BERTSubject(model_id="bert-base-uncased", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.10'})

model_registry[f'bert-base-uncased-layerwise'] = lambda: BERTGroup(model_id='bert-base-uncased', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names('bert-base-uncased')})