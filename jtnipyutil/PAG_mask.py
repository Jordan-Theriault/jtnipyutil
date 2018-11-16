import os
from jtnipyutil.fsmap import create_aqueduct_template
subj_list = [
    'sub-001',
    'sub-002',
    'sub-003',
    'sub-004',
    'sub-005',
    'sub-007',
    'sub-008',
    'sub-009',
    'sub-010',
    'sub-011',
    'sub-012',
    'sub-013',
#     'sub-014' # Missing - TODO, preprocess and add.
    'sub-015',
    'sub-016',
#     'sub-017', # Subject dropped out.
#     'sub-018' # High movement. DROP
    'sub-019',
#     'sub-020' # High movement. DROP
#     'sub-021' # High movement. DROP
    'sub-022',
    'sub-023',
    'sub-024',
    'sub-025',
    'sub-026',
#     'sub-027', # Subject dropped out.
#     'sub-028', # No stress task.
#     'sub-029', # Subject dropped out.
    'sub-030',
#     'sub-031' # High movement. DROP
    'sub-032',
#     'sub-033' # High movement. DROP
    'sub-034',
]
p_thresh_list = [95, 97.5, 99.9]
template = '/home/neuro/data/smooth/sub-*/model/sub-*/fwhm_1.5/data/sigmasquareds.nii.gz'
work_dir = '/home/neuro/workdir/PAG_mask'
space_mask =  '/home/neuro/atlases/FSMAP/PAG/search_region.nii'

create_aqueduct_template(subj_list, p_thresh_list, template, work_dir, space_mask)

# aq_temp_img = nib.Nifti1Image(aq_template, img_info.affine, img_info.header)
# nib.save(aq_temp_img, os.path.join(work_dir, 'MEAN_aqueduct_template.nii.gz'))
