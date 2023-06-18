import logging
import numpy as np
import os
import re
import scipy.io
import scipy.stats
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
import xarray as xr
from brainio.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from brainscore_language.utils.s3 import upload_data_assembly
SAVE_DIR='/om2/user/ehoseini/MyData/brain-score-language/dataset'

_logger = logging.getLogger(__name__)
is_sorted = lambda a: np.all(a[:-1] <= a[1:])

def load_ANNSet1_ECoG_full():
    data_dir = Path(f'/om/weka/evlab/ehoseini//MyData/ecog_DNN/crunched/ecog_ANNSET1_auditory_trainset_subs_10.pkl')
    data = pd.read_pickle(data_dir.__str__())
    atlas_data = dict()
    pair_signal_types = {'HighGamma_unipolar_gauss': 'elec_data_dec',
                         'HighGamma_unipolar_gauss_zscore': 'elec_data_zs_dec',
                         'HighGamma_unipolar_bandpass': 'envelope_dec',
                         'HighGamma_bipolar_gauss': 'bip_elec_data_dec',
                         'HighGamma_bipolar_gauss_zscore': 'bip_elec_data_zs_dec',
                         'HighGamma_bipolar_bandpass': 'bip_envelope_dec'}
    for id, signal_type_id in enumerate(pair_signal_types.keys()):
        data_atlas = data.sel(neuroid=(data.signal_type == pair_signal_types[signal_type_id]).values)
        # drop nonvalid electrodes
        data_atlas = data_atlas.sel(neuroid=data_atlas.electrode_valid.values.astype('bool'))
        data_atlas = data_atlas.sel(presentation=(data_atlas.stim_type == 'S').values)
        roi_list = [['non_language', 'language'][int(x)] for x in data_atlas.s_v_n.values]
        atlas_list = np.repeat(signal_type_id, data_atlas.shape[1])
        HCP_atlas = data_atlas.electrode_HCP_label.values
        HCP_atlas = ['N/A' if len(x) == 0 else x for x in (HCP_atlas)]
        assembly = NeuroidAssembly(data_atlas.values, coords={
            'experiment': ('presentation', np.repeat('ANNSet1_ECoG', data_atlas.shape[0])),
            'stimulus_num': ('presentation', data_atlas.stim_name.values),
            'stimulus_id': ('presentation', data_atlas.stim_id.values),
            'stimulus_value': ('presentation', data_atlas.stim_value.values),
            'stimulus': ('presentation', data_atlas.stimulus.values),
            'word_id': ('presentation', data_atlas.word_id.values),
            'key': ('presentation', data_atlas.key.values),
            'string': ('presentation', data_atlas.string.values),
            'list_id': ('presentation', data_atlas.list_id.values),
            'stim_type': ('presentation', data_atlas.stim_type.values),
            'stim_name': ('presentation', data_atlas.stim_name.values),
            'Trial_id': ('presentation', data_atlas.Trial_id.values),
            'Trial_abs_id': ('presentation', data_atlas.Trial_abs_id.values),
            # 'Trial_onset': ('presentation', data_atlas.Trial_onset.values),
            # 'Trial_abs_onset': ('presentation', data_atlas.Trial_abs_onset.values),
            'subject': ('neuroid', data_atlas.subject.values),
            'neuroid_id': ('neuroid', data_atlas.neuroid_id.values),
            'electrode_id': ('neuroid', data_atlas.electrode_valid.values),
            'electrode_valid': ('neuroid', data_atlas.electrode_valid.values),
            'electrode_label': ('neuroid', data_atlas.electrode_label.values),
            'electrode_posX_MNI': ('neuroid', np.stack(data_atlas.electrode_pos_mni.values, axis=0)[:, 0]),
            'electrode_posY_MNI': ('neuroid', np.stack(data_atlas.electrode_pos_mni.values, axis=0)[:, 1]),
            'electrode_posZ_MNI': ('neuroid', np.stack(data_atlas.electrode_pos_mni.values, axis=0)[:, 2]),
            'electrode_pos_HCP_atlas': ('neuroid', HCP_atlas),
            'S_vs_N_ratio': ('neuroid', data_atlas.s_v_n_ratio.values),
            'S_vs_N': ('neuroid', data_atlas.s_v_n.values),
            'roi': ('neuroid', roi_list),
            'filter_type': ('neuroid', atlas_list)
        }, dims=['presentation', 'neuroid'])

        assembly = assembly.sortby('Trial_abs_id')
        # make sure the word orde is not changed
        for grp_id, grp in assembly.groupby('Trial_abs_id'):
            assert is_sorted(grp.word_id.values)
        assembly.attrs['identifier'] = f"ANNSet1_ECoG.train.{signal_type_id}"
        atlas_data[signal_type_id] = assembly
        save_path = Path(SAVE_DIR, f"ANNSet1_ECoG.train.{signal_type_id}.pkl")
        with open(save_path.__str__(), 'wb') as f:
            pickle.dump(assembly, f)
        # save atlas data  in brainscore dir

    return atlas_data