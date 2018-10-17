import numpy as np
import nibabel as nib
import os
from jt_util import mask_img, clust_thresh, files_from_template
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
work_dir = '/home/neuro/workdir/PAG_mask'
PAG_region_mask =  '/home/neuro/atlases/FSMAP/PAG/search_region.nii'
subj_resid_template = '/home/neuro/data/smooth/sub-*/model/sub-*/fwhm_1.5/data/sigmasquareds.nii.gz'
thresh_list = [95, 97.5, 99.9]

# get all subjects—mask, threshold, cluster.

for subj in subj_list: # Get thresholded clusters.
try:
    img_file = nib.load(files_from_template(subj, os.path.join(work_dir, '*_aqueduct_template.nii.gz'))[0])
except:
    img_file  = files_from_template(subj, subj_resid_template)[0]
    img_info = nib.load(img_file)
    img = mask_img(img_file, PAG_region_mask, out_format = 'array') # loading done here. Slow.
    img = np.nanmean(img, axis=3) # Average data along time.
    for idx, thresh in enumerate(thresh_list):
        img_labeled = clust_thresh(img, cluster_k=[50,40,30], thresh = thresh)
        if idx == 0:
            all_labeled = img_labeled[..., np.newaxis]
        else:
            all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3) # stack thresholds along 4th dim.
    pag_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
    pag_img.header['cal_max'] = np.max(all_labeled) # fix header info
    pag_img.header['cal_min'] = 0 # fix header info
    nib.save(pag_img, os.path.join(work_dir, subj+'_aqueduct_template.nii.gz'))

## gather all subjects clusters/thresholds into a 5d array. ##########################################
for idx, subj in enumerate(subj_list):
    img_file = files_from_template(subj, os.path.join(work_dir, '*_aqueduct_template.nii.gz'))
    print(('getting data from %s') % img_file[0])
    img = nib.load(img_file[0]).get_data()
    if idx == 0:
        all_subj_data = img[..., np.newaxis]
    else:
        all_subj_data = np.append(all_subj_data, img[...,np.newaxis], axis=4)

## get mean across defaults: threshold (95) and cluster (1) ##########################################
aq_template = np.copy(all_subj_data[...,0,:])
aq_template[aq_template != 1] = 0
aq_template = np.mean(aq_template, axis=3)


while True:
    new_template = np.zeros(list(all_subj_data.shape[0:3]) + [all_subj_data.shape[-1]])
    for subj_idx, subj in enumerate(subj_list):
        corr_val = -101
        for thresh_idx, thresh in enumerate(thresh_list):
            if np.max(all_subj_data[..., thresh_idx, subj_idx]) > 0:
                for cluster in np.unique(all_subj_data[...,thresh_idx, subj_idx]):
                    if cluster == 0:
                        continue
                    else:
                        print(('checking sub %s, thresh %s, clust %s') % (subj, thresh, cluster))
                        test_array = np.copy(all_subj_data[...,thresh_idx, subj_idx]) # binarize array being tested.
                        test_array[test_array != cluster] = 0
                        test_array[test_array == cluster] = 1
                        clust_corr = np.correlate(np.ndarray.flatten(aq_template), # correlate with group mean.
                                                  np.ndarray.flatten(test_array))[0]
                        if clust_corr > corr_val:
                            print(('sub %s, thresh %s, clust %s, corr =  %s (prev max corr = %s)') %
                                  (subj, thresh, cluster, clust_corr, corr_val))
                            new_template[...,subj_idx] = test_array
                            corr_val = clust_corr
    if np.array_equal(np.around(aq_template, 4), np.around(np.mean(new_template, axis=3), 4)):
        print('We have converged on a stable average for aq_template.')
        break
    else:
        aq_template = np.mean(new_template, axis=3)
        print('new aq_template differs from previous iteration. Performing another iteration.')

    # End when new and old array are equal. Or, after so many iterations.



# aq_temp_img = nib.Nifti1Image(aq_template, img_info.affine, img_info.header)
# nib.save(aq_temp_img, os.path.join(work_dir, 'MEAN_aqueduct_template.nii.gz'))
