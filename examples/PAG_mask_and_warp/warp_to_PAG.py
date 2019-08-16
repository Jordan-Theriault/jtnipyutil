from jtnipyutil.fsmap import setup_DARTEL_warp_wf
import os
subj_list = ['sub-001','sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010']

data_templates = {'instructions': '/scratch/data/sub-*/model/sub-*/_modelestimate0/pe1_instructions.nii.gz',
                 'speech_prep': '/scratch/data/sub-*/model/sub-*/_modelestimate0/pe2_speech_prep.nii.gz',
                 'no_speech': '/scratch/data/sub-*/model/sub-*/_modelestimate0/pe3_no_speech.nii.gz'}
work_dir = '/scratch/wrkdir'
out_dir = '/scratch/output'
warp_template = os.path.join('/scratch/warp_templates', 'u_sub-*_pag_mask_Template.nii')

for data in data_templates:
    DARTEL_warp = setup_DARTEL_warp_wf(subj_list, data_templates[data],
                                       warp_template, os.path.join(work_dir, data), os.path.join(out_dir, data))
    DARTEL_warp.run(plugin='MultiProc')
