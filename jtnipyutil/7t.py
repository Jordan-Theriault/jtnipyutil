def create_aqueduct_template(subj_list, p_thresh_list, tempate, work_dir, region_mask):
    '''
    The workflow takes the following as input to wf.inputs.inputspec
    Input [Mandatory]:
        subject_list: list of subject IDs
            e.g. [sub-001, sub-002]

        p_thresh_list: list of floats representing p thresholds. Applied to resdiduals.
            e.g. [95, 97.5, 99.9]

        template: dictionary to identify all files.
            e.g. preproc_func_template = {'func': '/home/neuro/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'}
                The template can identify a larger set f files, and the subject_list will grab a subset.
                    e.g. The template may grab sub-001, sub-002, sub-003 ...
                    But if the subject_list only includes sub-001, then only sub-001 will be used.
                    This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

        work_dir: string, denoting path to working directoy.

        space_mask: string, denoting path to PAG search region mask.
    Output:

    TODO
    '''
    import nibabel as nib
    import os
    from jtnipyutil.util import files_from_template, clust_thresh, mask_img

    for subj in subj_list: # For each subjet, create aqueduct template file wtih all thresholded clusters.
        try:
            img_file = nib.load(files_from_template(subj, os.path.join(work_dir, '*_aqueduct_template.nii.gz'))[0])
        except:
            img_file  = files_from_template(subj, template)[0]
            img_info = nib.load(img_file)
            img = mask_img(img_file, space_mask, out_format = 'array') # loading done here. Slow.
            # img = np.nanmean(img, axis=3) # Average data along time.
            for thresh in p_thresh_list:
                img_labeled = clust_thresh(img, cluster_k=[50,40,30], thresh = thresh)
                if thresh = p_thresh_list[0]:
                    all_labeled = img_labeled[..., np.newaxis]
                else:
                    all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3) # stack thresholds along 4th dim.
            pag_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
            pag_img.header['cal_max'] = np.max(all_labeled) # fix header info
            pag_img.header['cal_min'] = 0 # fix header info
            nib.save(pag_img, os.path.join(work_dir, subj+'_aqueduct_template.nii.gz'))

    ## gather all subjects clusters/thresholds into a 5d array. ##########################################
    for subj in enumerate(subj_list):
        img_file = files_from_template(subj, os.path.join(work_dir, '*_aqueduct_template.nii.gz'))
        print(('getting data from %s') % img_file[0])
        img = nib.load(img_file[0]).get_data()
        if subj == subj_list[0]:
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
            for thresh_idx, thresh in enumerate(p_thresh_list):
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
