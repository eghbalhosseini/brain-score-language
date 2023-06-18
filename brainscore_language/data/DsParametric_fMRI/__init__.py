import logging

from brainscore_language import data_registry
from .data_packaging import  load_DsParametric_fMRI_V2


_logger = logging.getLogger(__name__)

BIBTEX = """
}"""
vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
vox_corr = {'language': (True, .1), 'auditory': (True, 0), 'visual': (True, 0)}

data_registry['DsParametric_fMRI.language_top_90'] = lambda: load_DsParametric_fMRI_V2(atlas='language',threshold=90)
data_registry['DsParametric_fMRI.language_top_80'] = lambda: load_DsParametric_fMRI_V2(atlas='language',threshold=80)
data_registry['DsParametric_fMRI.language_top_70'] = lambda: load_DsParametric_fMRI_V2(atlas='language',threshold=70)
