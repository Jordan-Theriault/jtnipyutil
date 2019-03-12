import os
from jtnipyutil.fsmap import create_aqueduct_template, make_PAG_masks, PAG_DARTEL
# from jtnipyutil.fsmap import create_aqueduct_template
subj_list = [
    'sub-001','sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010',
    'sub-011','sub-012','sub-013',
#     'sub-014' # Missing - TODO, preprocess and add.
    'sub-015','sub-016',
#     'sub-017', # Subject dropped out.
#     'sub-018' # High movement. DROP
    'sub-019',
#     'sub-020' # High movement. DROP
#     'sub-021' # High movement. DROP
    'sub-022','sub-023','sub-024','sub-025','sub-026',
#     'sub-027', # Subject dropped out.
#     'sub-028', # No stress task.
#     'sub-029', # Subject dropped out.
    'sub-030',
#     'sub-031' # High movement. DROP
    'sub-032',
#     'sub-033' # High movement. DROP
    'sub-034',
]
p_thresh_list = [95, 97.5, 99.9, 90]
# template = '/home/neuro/data/stress/smooth/sub-*/model/fwhm_1.5sub-*/_modelestimate0/sigmasquareds.nii.gz'
template = '/home/neuro/data/stress/nosmooth/sub-*/model/sub-*/_modelestimate0/sigmasquareds.nii.gz'
work_dir = '/home/neuro/workdir/PAG_mask'
space_mask =  '/home/neuro/atlases/FSMAP/PAG/search_region.nii'

create_aqueduct_template(subj_list, p_thresh_list, template, work_dir, space_mask)

data_template = os.path.join(work_dir, 'templates', '*_aqueduct_template.nii.gz')
gm_template = os.path.join(work_dir, 'gm', '*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')
y_minmax = [82,100]
z_minmax = [58,176]
make_PAG_masks(subj_list, data_template, gm_template, work_dir, y_minmax=y_minmax, z_minmax=z_minmax)


PAG_template = os.path.join(work_dir, 'pag_mask', '*_pag_mask.nii')
PAG_DARTEL = create_DARTEL_wf(subj_list, PAG_template, work_dir) # Creates DARTEL Template workflow.

PAG_DARTEL.run(plugin='MultiProc', plugin_args={'n_procs': 2})
# TODO - run this on the functional data as well. Modify code to allow this, as the list of dartel transforms and the data to be transformed are different.
