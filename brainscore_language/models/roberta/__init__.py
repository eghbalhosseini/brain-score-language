from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.roberta.modeling import RoBERTaSubject,RoBERTaGroup, get_layer_names


model_registry['roberta-base'] = lambda: RoBERTaSubject(model_id="roberta-base", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'roberta.encoder.layer.1'})


model_registry['roberta-base-layerwise'] = lambda: RoBERTaGroup(model_id="roberta-base", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names("roberta-base")})