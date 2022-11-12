from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.models.ctrl.modeling import CTRLSubject
from transformers import AutoModelForCausalLM
import re

model_id='ctrl'
model_registry[model_id] = lambda: CTRLSubject(model_id=model_id, region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.46'})
