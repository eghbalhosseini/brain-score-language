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
def vox_reliability_Kell2018(assembly=None):
    for sub_id,sub_dat in assembly.groupby('subject'):
        sub_sess1=sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 1)
        sub_sess2 = sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 2)
        plt.hist(sub_sess1.repetition_corr.values,50)
        plt.hist(sub_sess2.repetition_corr.values,50)
        plt.show()
        for vox_id, vox_dat in sub_dat.groupby('voxel_num'):
            True
    NotImplementedError

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
    data_dir=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ANNsent_trainset_subs_8_thresh-{threshold}.pkl')
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
            'stimulus_num':('presentation',data_atlas.stim_name.values),
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

def load_ANNSet1_fMRI_extended(threshold=80):
    data_dir=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ANNsent_all_subs_16_thresh-{threshold}_analyzed.pkl')
    data=pd.read_pickle(data_dir.__str__())
    atlas_data=dict()
    name_pair={'language':'network_lang','auditory':'network_aud','visual':'network_vis'}
    for id, atlas_id in enumerate(['language','auditory','visual']):
        data_atlas=data.sel(voxels=data[name_pair[atlas_id]].values)
        sentence_dict=dict(ChainMap(*data_atlas.stim_value.values))
        #[list(x.values()) for x in data_atlas.stim_value.values]
        sentences=[sentence_dict[str(stim_val.values)] for stim_val in data_atlas.stim_name]
        #fix issue with sess_id
        all_sub_sessions=[]
        for sub_id, sub_dat in data_atlas.groupby('subject'):
            True
            sub_sess1 = sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 1)
            sub_sess1_ave=sub_sess1.mean('repeat')
            sub_sess1_ave=sub_sess1_ave.assign_coords({'sess_id':('voxels',sub_sess1.sess_id.values[:,0].squeeze())})
            sub_sess2 = sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 2)
            sub_sess2_ave = sub_sess2.mean('repeat')
            sub_sess2_ave = sub_sess2_ave.assign_coords(
                {'sess_id': ('voxels', sub_sess2.sess_id.values[:, 0].squeeze())})
            sub_sess=xr.concat([sub_sess1_ave,sub_sess2_ave],'voxels')
            assert(sub_sess.shape[1]==(sub_sess1_ave.shape[1]+sub_sess2_ave.shape[1]))
            all_sub_sessions.append(sub_sess)

        all_sub_sessions_xr=xr.concat(all_sub_sessions,'voxels')
        assert (all_sub_sessions_xr.shape[1] == np.sum([x.shape[1] for x in all_sub_sessions]))
        roi_list = all_sub_sessions_xr[name_pair[atlas_id]+'_rois'].values
        atlas_list=np.repeat(atlas_id,all_sub_sessions_xr.shape[1])
        assembly=NeuroidAssembly(all_sub_sessions_xr.values,coords={
            'experiment':('presentation',np.repeat('ANNSet1_fMRI',all_sub_sessions_xr.shape[0])),
            'stimulus_num':('presentation',all_sub_sessions_xr.stim_name.values),
            'stimulus_id': ('presentation', all_sub_sessions_xr.stim_id.values),
            'sentence':('presentation',sentences),
            'stimulus': ('presentation', sentences),
            'list_id': ('presentation', all_sub_sessions_xr.list_id.values),

            'stim_type': ('presentation', all_sub_sessions_xr.stim_type.values),
            'stim_name': ('presentation', all_sub_sessions_xr.stim_name.values),
            'Trial_id': ('presentation', all_sub_sessions_xr.Trial_id.values),
            'TR_onset': ('presentation', all_sub_sessions_xr.TR_onset.values),
            'TR_recorded': ('presentation', all_sub_sessions_xr.TR_recorded.values),
            'TR_duration': ('presentation', all_sub_sessions_xr.TR_duration.values),
            'subject': ('neuroid',all_sub_sessions_xr.subject.values),
            'sess_id': ('neuroid', all_sub_sessions_xr.sess_id.values),
            'tval_lang': ('neuroid', all_sub_sessions_xr.tval_lang.values),
            'neuroid_id':('neuroid',all_sub_sessions_xr.neuroid_id.values),
            'voxel_num':('neuroid',all_sub_sessions_xr.voxel_num.values),
            'repetition_corr_ratio': ('neuroid', all_sub_sessions_xr.repetition_corr_ratio.values),
            'repetition_corr': ('neuroid', all_sub_sessions_xr.repetition_corr.values),
            'roi':('neuroid',roi_list),
            'atlas':('neuroid',atlas_list)
        },dims=['presentation','neuroid'])
        assembly = assembly.sortby('stimulus_id')
        assembly.attrs['identifier'] = f"ANNSet1_fMRI.all.{atlas_id}_top_{threshold}_V2"
        atlas_data[atlas_id]=assembly
    return atlas_data


def load_ANNSet1_fMRI_for_RSA_extended(threshold=80):
    data_dir=Path(f'/om/weka/evlab/ehoseini//MyData/fmri_DNN/outputs/ANNsent_all_subs_16_thresh-{threshold}_analyzed.pkl')
    data=pd.read_pickle(data_dir.__str__())
    atlas_data=dict()
    name_pair={'language':'network_lang','auditory':'network_aud','visual':'network_vis'}
    for id, atlas_id in enumerate(['language','auditory','visual']):
        data_atlas=data.sel(voxels=data[name_pair[atlas_id]].values)
        sentence_dict=dict(ChainMap(*data_atlas.stim_value.values))
        #[list(x.values()) for x in data_atlas.stim_value.values]
        sentences=[sentence_dict[str(stim_val.values)] for stim_val in data_atlas.stim_name]
        #fix issue with sess_id
        all_sub_sessions=[]
        for sub_id, sub_dat in data_atlas.groupby('subject'):
            True
            sub_sess1 = sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 1)
            sub_sess1=sub_sess1.assign_coords({'sess_id':('voxels',sub_sess1.sess_id.values[:,0].squeeze())})
            sub_sess2 = sub_dat.sel(voxels=sub_dat.sess_id[:, 0] == 2)

            sub_sess2 = sub_sess2.assign_coords(
                {'sess_id': ('voxels', sub_sess2.sess_id.values[:, 0].squeeze())})
            sub_sess=xr.concat([sub_sess1,sub_sess2],'voxels')
            assert(sub_sess.shape[2]==(sub_sess1.shape[2]+sub_sess2.shape[2]))
            all_sub_sessions.append(sub_sess)

        all_sub_sessions_xr=xr.concat(all_sub_sessions,'voxels')
        assert (all_sub_sessions_xr.shape[2] == np.sum([x.shape[2] for x in all_sub_sessions]))
        roi_list = all_sub_sessions_xr[name_pair[atlas_id]+'_rois'].values
        atlas_list=np.repeat(atlas_id,all_sub_sessions_xr.shape[2])
        assembly=NeuroidAssembly(all_sub_sessions_xr.values,coords={
            'experiment':('presentation',np.repeat('ANNSet1_fMRI',all_sub_sessions_xr.shape[1])),
            'stimulus_num':('presentation',all_sub_sessions_xr.stim_name.values),
            'stimulus_id': ('presentation', all_sub_sessions_xr.stim_id.values),
            'sentence':('presentation',sentences),
            'stimulus': ('presentation', sentences),
            'list_id': ('presentation', all_sub_sessions_xr.list_id.values),
            'stim_type': ('presentation', all_sub_sessions_xr.stim_type.values),
            'stim_name': ('presentation', all_sub_sessions_xr.stim_name.values),
            'Trial_id': ('presentation', all_sub_sessions_xr.Trial_id.values),
            'TR_onset': ('presentation', all_sub_sessions_xr.TR_onset.values),
            'TR_recorded': ('presentation', all_sub_sessions_xr.TR_recorded.values),
            'TR_duration': ('presentation', all_sub_sessions_xr.TR_duration.values),
            'subject': ('neuroid',all_sub_sessions_xr.subject.values),
            'sess_id': ('neuroid', all_sub_sessions_xr.sess_id.values),
            'tval_lang': ('neuroid', all_sub_sessions_xr.tval_lang.values),
            'neuroid_id':('neuroid',all_sub_sessions_xr.neuroid_id.values),
            'voxel_num':('neuroid',all_sub_sessions_xr.voxel_num.values),
            'repetition_corr_ratio': ('neuroid', all_sub_sessions_xr.repetition_corr_ratio.values),
            'repetition_corr': ('neuroid', all_sub_sessions_xr.repetition_corr.values),
            'roi':('neuroid',roi_list),
            'atlas':('neuroid',atlas_list),
            'sess_repeat':('repeat',all_sub_sessions_xr.sess_repeat.values),
            #'abs_repeat': ('repeat', all_sub_sessions_xr.abs_repeat.values)
        },dims=['repeat','presentation','neuroid'])
        assembly = assembly.sortby('stimulus_id')
        assembly.attrs['identifier'] = f"ANNSet1_fMRI_RSA.all.{atlas_id}_top_{threshold}_V2"
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


def load_ANNSet1_fMRI_V2(atlas=None,group_id='best',threshold=70):
    atlas_list=load_ANNSet1_fMRI_extended(threshold)
    ## preset for voxel reliability, and correlation
    #s_v_n = {'language': (False,0.95), 'auditory': (False,0), 'visual': (False,0)}
    vox_reliability = {'language': (True, .95), 'auditory': (True, .95), 'visual': (True, .95)}
    vox_corr = {'language': (True, 0), 'auditory': (True, 0), 'visual': (True, 0)}
    groups = {'train':[('837',2),('682',1),('848',1),('865',1),
                       ('906',2),('913', 2),('916', 1),('880', 2)],
              'test': [('837', 1),('682', 2),('848', 2),('865', 2),
                        ('906', 1),('913', 1),('916', 2),('880', 1)],
              'best': [ ('682', 1),('837', 1), ('848', 2), ('865', 1), ('880', 1)
                       ,('906', 1), ('913', 1), ('916', 1)]
              }

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

    vox_selection=np.logical_and(vox_corr_vec,vox_rel_vec)
    #
    atlas_out = atlas_out.sel(neuroid=vox_selection)
    sub_sess_pairs=groups[group_id]
    sub_sess_out=[]
    for sub_id,sess_id in sub_sess_pairs:
        subj_=atlas_out.sel(neuroid=(atlas_out.subject.values==sub_id))
        sub_sess_=subj_.sel(neuroid=(subj_.sess_id.values==sess_id))
        print(f'{sub_id},{sess_id}, {sub_sess_.shape[1]}')
        sub_sess_out.append(sub_sess_)

    sub_sess_out=xr.concat(sub_sess_out,dim='neuroid')

    sub_sess_out.attrs['voxel_selection'] = {'voxel_reliability': vox_reliability[atlas],
                                          'voxel_correlation': vox_corr[atlas]}
    sub_sess_out.attrs['functional_threshold'] = threshold
    sub_sess_out.attrs['group_selection'] = {'name':group_id,'group_info':groups[group_id]}

    #atlas_out.attrs['voxel_selection']={'voxel_reliability':vox_reliability[atlas],'voxel_correlation':vox_corr[atlas]}
    #atlas_out.attrs['functional_threshold']=threshold

    return sub_sess_out


def load_ANNSet1_fMRI_RSA_V2(atlas=None,group_id='train',threshold=90):
    atlas_list=load_ANNSet1_fMRI_for_RSA_extended(threshold)
    ## preset for voxel reliability, and correlation
    #s_v_n = {'language': (False,0.95), 'auditory': (False,0), 'visual': (False,0)}
    vox_reliability = {'language': (True, .95), 'auditory': (True, .95), 'visual': (True, .95)}
    vox_corr = {'language': (True, 0), 'auditory': (True, 0), 'visual': (True, 0)}
    groups = {'train':[('837',2),('682',1),('848',1),('865',1),
                       ('906',2),('913', 2),('916', 1),('880', 2)],
              'test': [('837', 1),('682', 2),('848', 2),('865', 2),
                        ('906', 1),('913', 1),('916', 2),('880', 1)],
              'best': [ ('682', 1),('837', 1), ('848', 2), ('865', 1), ('880', 1)
                       ,('906', 1), ('913', 1), ('916', 1)]
              }

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

    vox_selection=np.logical_and(vox_corr_vec,vox_rel_vec)
    #
    atlas_out = atlas_out.sel(neuroid=vox_selection)
    sub_sess_pairs=groups[group_id]
    sub_sess_out=[]
    for sub_id,sess_id in sub_sess_pairs:
        subj_=atlas_out.sel(neuroid=(atlas_out.subject.values==sub_id))
        sub_sess_=subj_.sel(neuroid=(subj_.sess_id.values==sess_id))
        print(f'{sub_id},{sess_id}, {sub_sess_.shape[1]}')
        sub_sess_out.append(sub_sess_)


    sub_sess_xr=xr.concat(sub_sess_out,dim='neuroid')
    assert (sub_sess_xr.shape[2] == np.sum([x.shape[2] for x in sub_sess_out]))

    sub_sess_xr.attrs['voxel_selection'] = {'voxel_reliability': vox_reliability[atlas],
                                          'voxel_correlation': vox_corr[atlas]}
    sub_sess_xr.attrs['functional_threshold'] = threshold
    sub_sess_xr.attrs['group_selection'] = {'name':group_id,'group_info':groups[group_id]}

    #atlas_out.attrs['voxel_selection']={'voxel_reliability':vox_reliability[atlas],'voxel_correlation':vox_corr[atlas]}
    #atlas_out.attrs['functional_threshold']=threshold

    return sub_sess_xr

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #upload_ANNSet1_fMRI(atlas='language')
    #upload_ANNSet1_fMRI(atlas='auditory')
    #upload_ANNSet1_fMRI(atlas='visual')