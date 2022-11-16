from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.xlnet.modeling import XLNetSubject,XLNetGroup, get_layer_names

model_registry['xlnet-large-cased'] = lambda: XLNetSubject(model_id="xlnet-large-cased", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.layer.23'})


model_registry['xlnet-large-cased-layerwise'] = lambda: XLNetGroup(model_id="xlnet-large-cased", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names("xlnet-large-cased")})