from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.xlm.modeling import XLMSubject, XLMGroup, get_layer_names






model_registry['xlm-mlm-en-2048'] = lambda: XLMSubject(model_id="xlm-mlm-en-2048", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.layer_norm2.11'})


model_registry['xlm-mlm-en-2048-layerwise'] = lambda: XLMGroup(model_id="xlm-mlm-en-2048", region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names("xlm-mlm-en-2048")})