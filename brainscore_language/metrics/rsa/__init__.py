from brainscore_language import metric_registry
from .metric import rsa_correlation

metric_registry['rsa_correlation'] = rsa_correlation
