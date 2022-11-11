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
import xarray as xr
from brainio.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from brainscore_language.utils.s3 import upload_data_assembly
from collections import ChainMap
_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""

def ANNSet1_fMRI_auditory():
    return load_ANNSet1_fMRI(atlas='auditory')

def ANNSet1_fMRI_language():
    return load_ANNSet1_fMRI(atlas='language',threshold=70)

def ANNSet1_fMRI_visual():
    return load_ANNSet1_fMRI(atlas='visual',threshold=70)


def upload_ANNSet1_fMRI(atlas):
    assembly = load_ANNSet1_fMRI(atlas=atlas)
    upload_data_assembly(assembly,
                         assembly_identifier=f"ANNSet1_fMRI.{atlas}")


def load_ANNSet1_fMRI_full(threshold=80):
    data_dir=Path(f'/om/user/ehoseini/MyData/fmri_DNN/outputs/ANNsent_trainset_subs_8_thresh-{threshold}.pkl')
    data=pd.read_pickle(data_dir.__str__())
    atlas_data=dict()
    name_pair={'language':'network_lang','auditory':'network_aud','visual':'network_vis'}
    for id, atlas_id in enumerate(['language','auditory','visual']):
        data_atlas=data.sel(voxels=data[name_pair[atlas_id]].values)
        sentence_dict=dict(ChainMap(*data_atlas.stim_value.values))
        sentences=[sentence_dict[str(stim_val.values)] for stim_val in data_atlas.stim_name]
        # select reliable voxels
        roi_list = data_atlas[name_pair[atlas_id]+'_rois'].values
        atlas_list=np.repeat(atlas_id,data_atlas.shape[1])
        assembly=NeuroidAssembly(data_atlas.values,coords={
            'experiment':('presentation',np.repeat('ANNSet1_fMRI',data_atlas.shape[0])),
            'stimulus_num':('pre2sentation',data_atlas.stim_name.values),
            'stimulus_id': ('presentation', data_atlas.stim_id.values),
            'sentence':('presentation',sentences),
            'stimulus': ('presentation', sentences),
            'list_id': ('presentation', data_atlas.list_id.values),
            'stim_type': ('presentation', data_atlas.stim_type.values),
            'stim_name': ('presentation', data_atlas.stim_name.values),
            'Trial_id': ('presentation', data_atlas.Trial_id.values),
            'TR_onset': ('presentation', data_atlas.TR_onset.values),
            'TR_recorded': ('presentation', data_atlas.TR_recorded.values),
            'TR_duration': ('presentation', data_atlas.TR_duration.values),
            'subject': ('neuroid',data_atlas.subject.values),
            'neuroid_id':('neuroid',data_atlas.neuroid_id.values),
            'voxel_num':('neuroid',data_atlas.voxel_num.values),
            'S_vs_N_ratio': ('neuroid', data_atlas.S_vs_N_ratio.values),
            'repetition_corr_ratio': ('neuroid', data_atlas.repetition_corr_ratio.values),
            'repetition_corr': ('neuroid', data_atlas.repetition_corr.values),
            'roi':('neuroid',roi_list),
            'atlas':('neuroid',atlas_list)
        },dims=['presentation','neuroid'])
        assembly = assembly.sortby('stimulus_id')
        assembly.attrs['identifier'] = f"ANNSet1_fMRI.train.{atlas_id}_top_{threshold}"
        atlas_data[atlas_id]=assembly
    return atlas_data


def load_ANNSet1_fMRI(atlas=None,threshold=70):
    atlas_list=load_ANNSet1_fMRI_full(threshold)
    ## preset for voxel reliability, and correlation
    s_v_n = {'language': (False,0.95), 'auditory': (False,0), 'visual': (False,0)}
    vox_reliability = {'language': (True, .95), 'auditory': (True, .95), 'visual': (True, .95)}
    vox_corr = {'language': (True, .1), 'auditory': (True, .1), 'visual': (True, .1)}
    atlas_out=atlas_list[atlas]
    ## subselect based on reliability
    if vox_corr[atlas][0]:
        vox_corr_vec=(atlas_out.repetition_corr>vox_corr[atlas][1]).values
    else:
        vox_corr_vec = (atlas_out.repetition_corr > -np.inf).values

    if vox_reliability[atlas][0]:
        vox_rel_vec = (atlas_out.repetition_corr_ratio > vox_reliability[atlas][1]).values
    else:
        vox_rel_vec = (atlas_out.repetition_corr_ratio > -np.inf).values

    if s_v_n[atlas][0]:
        vox_svn_vec = (atlas_out.S_vs_N_ratio > s_v_n[atlas][1]).values
    else:
        vox_svn_vec = (atlas_out.S_vs_N_ratio > -np.inf).values
    vox_selection=np.logical_and(np.logical_and(vox_corr_vec,vox_rel_vec),vox_svn_vec)
    #
    atlas_out = atlas_out.sel(neuroid=vox_selection)
    atlas_out.attrs['voxel_selection']={'s_v_n':s_v_n[atlas],'voxel_reliability':vox_reliability[atlas],'voxel_correlation':vox_corr[atlas]}
    atlas_out.attrs['functional_threshold']=threshold

    #import matplotlib.pyplot as plt
    #plt.hist(atlas_out.repetition_corr.values,100)
    #plt.show()
    #plt.hist(atlas_out.S_vs_N_ratio.values, 50)
    #plt.show()
    #plt.hist(atlas_out.repetition_corr_ratio.values, 50)
    #plt.show()
    return atlas_out


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #upload_ANNSet1_fMRI(atlas='language')
    #upload_ANNSet1_fMRI(atlas='auditory')
    #upload_ANNSet1_fMRI(atlas='visual')