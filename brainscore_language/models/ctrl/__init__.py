from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.ctrl.modeling import CTRLSubject, CTRLGroup, get_layer_names


model_registry['ctrl'] = lambda: CTRLSubject(model_id='ctrl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.46'})


model_registry['ctrl-layerwise'] = lambda: CTRLGroup(model_id='ctrl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names('ctrl')})
