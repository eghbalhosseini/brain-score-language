from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.albert.modeling import ALBERTSubject,ALBERTGroup, get_layer_names

model_registry['albert-xxlarge-v2'] = lambda: ALBERTSubject(model_id="albert-xxlarge-v2", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'encoder.albert_layer_groups.3'})


model_registry['albert-xxlarge-v2-layerwise'] = lambda: ALBERTGroup(model_id="albert-xxlarge-v2", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names("albert-xxlarge-v2")})