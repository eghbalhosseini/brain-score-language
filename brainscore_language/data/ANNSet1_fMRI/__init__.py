import logging

from brainscore_language import data_registry
from .data_packaging import load_ANNSet1_fMRI, load_ANNSet1_fMRI_V2,load_ANNSet1_fMRI_RSA_V2


_logger = logging.getLogger(__name__)

BIBTEX = """
}"""
data_registry['ANNSet1_fMRI.train.language_top_90'] = lambda: load_ANNSet1_fMRI(atlas='language',threshold=90)
data_registry['ANNSet1_fMRI.train.language_top_80'] = lambda: load_ANNSet1_fMRI(atlas='language',threshold=80)
data_registry['ANNSet1_fMRI.train.language_top_70'] = lambda: load_ANNSet1_fMRI(atlas='language',threshold=70)
data_registry['ANNSet1_fMRI.train.auditory'] = lambda: load_ANNSet1_fMRI(atlas='auditory',threshold=70)
data_registry['ANNSet1_fMRI.train.visual'] = lambda: load_ANNSet1_fMRI(atlas='visual',threshold=70)

data_registry['ANNSet1_fMRI_RSA.train.language_top_90'] = lambda: load_ANNSet1_fMRI_RSA_V2(atlas='language',threshold=90)
data_registry['ANNSet1_fMRI_RSA.train.language_top_80'] = lambda: load_ANNSet1_fMRI_RSA_V2(atlas='language',threshold=80)
data_registry['ANNSet1_fMRI_RSA.train.language_top_70'] = lambda: load_ANNSet1_fMRI_RSA_V2(atlas='language',threshold=70)
data_registry['ANNSet1_fMRI_RSA.train.auditory'] = lambda: load_ANNSet1_fMRI_RSA_V2(atlas='auditory',threshold=70)
data_registry['ANNSet1_fMRI_RSA.train.visual'] = lambda: load_ANNSet1_fMRI_RSA_V2(atlas='visual',threshold=70)



data_registry['ANNSet1_fMRI.best.language_top_90_V2'] = lambda: load_ANNSet1_fMRI_V2(atlas='language',group_id='best',threshold=90)
data_registry['ANNSet1_fMRI.best.language_top_80_V2'] = lambda: load_ANNSet1_fMRI_V2(atlas='language',group_id='best',threshold=80)
data_registry['ANNSet1_fMRI.best.language_top_70_V2'] = lambda: load_ANNSet1_fMRI_V2(atlas='language',group_id='best',threshold=70)
data_registry['ANNSet1_fMRI.best.auditory_V2'] = lambda: load_ANNSet1_fMRI_V2(atlas='auditory',group_id='best',threshold=70)
data_registry['ANNSet1_fMRI.best.visual_V2'] = lambda: load_ANNSet1_fMRI_V2(atlas='visual',group_id='best',threshold=70)