import numpy as np
from brainscore_language import load_dataset

def test_data():
    assembly = load_dataset('ANNSet1_fMRI.train.language_top_90')
    assert len(assembly['presentation']) == 200
    assert 'Most ships perform this drill before even leaving the dock.' in assembly['stimulus'].values
    assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation']), "not all stimulus_ids unique"
    assert set(assembly['roi'].values) == {'lang_LH_AngG','lang_LH_AntTemp','lang_LH_IFG','lang_LH_IFGorb','lang_LH_MFG',
                                           'lang_LH_PostTemp','lang_RH_AngG','lang_RH_AntTemp','lang_RH_IFG',
                                            'lang_RH_IFGorb','lang_RH_MFG','lang_RH_PostTemp'}
    mean_assembly = assembly.groupby('subject').mean()
    assert not np.isnan(mean_assembly).any(), "Each stimulus should have at least one subject's data"
