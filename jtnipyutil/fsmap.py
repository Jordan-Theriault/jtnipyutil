def create_aqueduct_template(subj_list, p_thresh_list, template, work_dir, space_mask):
    '''
    The workflow takes the following as input to wf.inputs.inputspec
    Input [Mandatory]:
        subj_list: list of subject IDs
            e.g. [sub-001, sub-002]

        p_thresh_list: list of floats representing p thresholds. Applied to resdiduals.
            e.g. [95, 97.5, 99.9]

        template: string to identify all PAG aqueduct files (using glob).
            e.g. preproc_func_template = '/home/neuro/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
                The template can identify a larger set f files, and the subject_list will grab a subset.
                    e.g. The template may grab sub-001, sub-002, sub-003 ...
                    But if the subject_list only includes sub-001, then only sub-001 will be used.
                    This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

        work_dir: string, denoting path to working directory.

        space_mask: string, denoting path to PAG search region mask.
    Output:

    TODO
    '''
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import os
    from jtnipyutil.util import files_from_template, clust_thresh, mask_img

    for subj in subj_list: # For each subjet, create aqueduct template file wtih all thresholded clusters.
        try:
            img_file = nib.load(files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
        except:
            img_file  = files_from_template(subj, template)[0]
            img_info = nib.load(img_file)
            img = mask_img(img_file, space_mask, out_format = 'array') # loading done here. Slow.
            # img = np.nanmean(img, axis=3) # Average data along time.
            for thresh in p_thresh_list:
                img_labeled = clust_thresh(img, cluster_k=[50], thresh = thresh)
                if thresh == p_thresh_list[0]:
                    all_labeled = img_labeled[..., np.newaxis]
                else:
                    all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3) # stack thresholds along 4th dim.
            pag_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
            pag_img.header['cal_max'] = np.max(all_labeled) # fix header info
            pag_img.header['cal_min'] = 0 # fix header info
            try:
                nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
            except:
                os.makedirs(os.path.join(work_dir, 'subj_clusts'))
                nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))

    ## gather all subjects clusters/thresholds into a 5d array. ##########################################
    for subj in subj_list:
        img_file = files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))
        print(('getting data from %s') % img_file[0])
        img = nib.load(img_file[0]).get_data()
        if subj == subj_list[0]:
            all_subj_data = img[..., np.newaxis]
        else:
            all_subj_data = np.append(all_subj_data, img[...,np.newaxis], axis=4)

    ## get mean across defaults: threshold (95) and cluster (1) ##########################################
    # This establishes a template to judge which threshold fits it best.
    # Average is across all subjects.
    aq_template = np.copy(all_subj_data[...,0,:])
    aq_template[aq_template != 1] = 0
    aq_template = np.mean(aq_template, axis=3)
    # set up report.
    aq_report = pd.DataFrame(columns=['sub', 'thresh', 'clust', 'corr', 'iter', 'FLAG'], data={'sub':subj_list, 'iter':[0]*len(subj_list), 'FLAG':['']*len(subj_list)})
    aq_report = aq_report.set_index('sub')
    while True:
        aq_report['iter'] = aq_report['iter'] + 1
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
                            clust_corr = np.corrcoef(np.ndarray.flatten(aq_template), # correlate with group mean.
                                                      np.ndarray.flatten(test_array))[0,1]
                            if clust_corr > corr_val:
                                print(('sub %s, thresh %s, clust %s, corr =  %s (prev max corr = %s)') %
                                      (subj, thresh, cluster, clust_corr, corr_val))
                                aq_report.at[subj, 'thresh'] = thresh
                                aq_report.at[subj, 'clust'] = cluster
                                aq_report.at[subj, 'corr'] = clust_corr
                                if clust_corr < .3:
                                    aq_report.at[subj, 'FLAG'] = 'CHECK'
                                else:
                                    aq_report.at[subj, 'FLAG'] = ''
                                new_template[...,subj_idx] = test_array
                                corr_val = clust_corr

        if np.array_equal(np.around(aq_template, 4), np.around(np.mean(new_template, axis=3), 4)):
            print('We have converged on a stable average for aq_template.')
            break
        else:
            aq_template = np.mean(new_template, axis=3)
            print('new aq_template differs from previous iteration. Performing another iteration.')

    for img_idx in range(0, new_template.shape[-1]):
        print(('Saving aqueduct for %s') % (subj_list[img_idx]))
        subj_temp = nib.Nifti1Image(new_template[...,img_idx], img_info.affine, img_info.header)
        subj_temp.header['cal_max'] = 1 # fix header info
        subj_temp.header['cal_min'] = 0 # fix header info
        try:
            nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
        except:
            os.makedirs(os.path.join(work_dir, 'templates'))
            nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))

    print('Saving aqueduct mean template.')
    aq_temp_img = nib.Nifti1Image(aq_template, img_info.affine, img_info.header)
    aq_temp_img.header['cal_max'] = 1 # fix header info
    aq_temp_img.header['cal_min'] = 0 # fix header info
    nib.save(aq_temp_img, os.path.join(work_dir, 'templates', 'MEAN_aqueduct_template.nii.gz'))

    print('Saving report')
    aq_report.to_csv(os.path.join(work_dir, 'templates', 'report.csv'))

def make_PAG_masks(subj_list, data_template, gm_template, work_dir, gm_thresh = .5, gm_spline=3, dilation_r=2, x_minmax=False, y_minmax=False, z_minmax=False):
    '''
    subj_list: list of subject IDs
        e.g. [sub-001, sub-002]

    data_template: string to identify all PAG aqueduct files (using glob).
        e.g. data_template = os.path.join(work_dir, 'templates', '*_aqueduct_template.nii.gz')
            The template can identify a larger set of files, and the subject_list will grab a subset.
                e.g. The template may grab sub-001, sub-002, sub-003 ...
                But if the subject_list only includes sub-001, then only sub-001 will be used.
                This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

    gm_template: string to identify all PAG aqueduct files (using glob).
        e.g. gm_template = os.path.join('work_dir, 'gm', '*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')

    work_dir: string, denoting path to working directory.

    gm_thresh: float specifying the probability to threshold gray matter mask.
        Default: .5

    gm_spline: integer specifying the spline order to use to reslice gray matter to native space (if necessary)
        Default: 3

    dilation_r: integer specifying the number of voxels to dilate the aqueduct.
        Default: 2

    x_minmax: list of 2 integers, denoting min and max X voxels to include in mask.
        e.g. [0,176]. Defaults to full range of PAG aqueduct image.

    y_minmax: list of 2 integers, denoting min and max Y voxels to include in mask.
            e.g. [82,100]. Defaults to full range of PAG aqueduct image.

    z_minmax: list of 2 integers, denoting min and max Z voxels to include in mask.
        e.g. [58,176]. Defaults to full range of PAG aqueduct image.

    '''
    import nibabel as nib
    import numpy as np
    from skimage.transform import resize
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import ball
    import os
    from jtnipyutil.util import files_from_template, clust_thresh, mask_img

    for subj in subj_list:
        # get aqueduct.
        img_file = nib.load(files_from_template(subj, data_template)[0])
        img = img_file.get_data()
        # get gray matter, binarize at threshold.
        gm_file = nib.load(files_from_template(subj, gm_template)[0])
        gm_img = gm_file.get_data()
        if img.shape[0:3] !=  gm_img.shape[0:3]:
            gm_img = resize(gm_img, img.shape[0:3], order=gm_spline, preserve_range=True)
        gm_img[np.where(gm_img < gm_thresh)] = 0
        # create mask for PAG location.
        if not x_minmax:
            x_minmax = [0,list(img.shape)[0]]
        if not y_minmax:
            y_minmax = [0,list(img.shape)[1]]
        if not z_minmax:
            z_minmax = [0,list(img.shape)[2]]
        loc_mask = np.zeros(list(img.shape))
        loc_mask[x_minmax[0]:x_minmax[1],
             y_minmax[0]:y_minmax[1],
             z_minmax[0]:z_minmax[1]] = 1
        # dilate and subtract original aqueduct.
        pag = binary_dilation(img, ball(dilation_r)).astype(img.dtype) - img
        pag = pag*gm_img # multiply by thresholded gm probability mask.
        pag = pag*loc_mask # threshold by general PAG location cutoffs.
        pag_file = nib.Nifti1Image(pag, img_file.affine, img_file.header)
        try:
            nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii.gz'))
        except:
            os.makedirs(os.path.join(work_dir, 'pag_mask'))
            nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii.gz'))

def PAG_DARTEL(subj_list, PAG_template,
               it_params=[(3, (4, 2, 1e-06), 1, 16),
                          (3, (2, 1, 1e-06), 1, 8),
                          (3, (1, 0.5, 1e-06), 2, 4),
                          (3, (0.5, 0.25, 1e-06), 4, 2),
                          (3, (0.25, 0.125, 1e-06), 16, 1),
                          (3, (0.25, 0.125, 1e-06), 64, 0.5)],
               opt_params=(0.01, 3, 3),
               reg_form='Linear', b_spline=4, warp_iter=6, fwhm=[0,0,0]):
    '''
    Aligns all PAG images to a template (average of all images), then warps images into MNI space (using an SPM tissue probability map, see https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf, section 25.4).

    subj_list: list of subject IDs
        e.g. [sub-001, sub-002]

    PAG_template: string to identify all PAG files (using glob).
        e.g. PAG_template = TODO
            The template can identify a larger set of files, and the subject_list will grab a subset.
                e.g. The template may grab sub-001, sub-002, sub-003 ...
                But if the subject_list only includes sub-001, then only sub-001 will be used.
                This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

    it_params:
        List 3 to 12 tuples for each iteration
         - Inner iterations: 1 <= a long integer <= 10
         - Regularization parameters: a tuple of the form: (a float, a float, a float)
         - Time points for deformation model: 1, 2, 4, 8, 16, 32, 64, 128, 256, or 512
         - smoothing parameter: 0, 0.5, 1, 2, 4, 8, 16, 32
            DARTEL iteration parameters

    opt_params: (a tuple of the form:
             - LM regularization: a float
             - cycles of multigrid solver: 1 <= a long integer <= 8
             - relaxation iterations: 1 <= a long integer <= 8
         DARTEL optimization parameters

    reg_form: ('Linear' or 'Membrane' or 'Bending')
        DARTEL: Form of regularization energy term
    '''
    import nibabel as nib
    import numpy as np
    from nipype.interfaces.spm.process import DARTEL, DARTELNORM2MNI, CreateWarped
    import nipype.pipeline.engine as pe
    import os
    from jtnipyutil.util import files_from_template
    # set up workflow.
    DARTEL_wf = pe.Workflow(name='DARTEL_wf')
    DARTEL_wf.base_dir = work_dir

    # set up sinker
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')

    # get images
    PAG_images = files_from_template(subj_list, PAG_template)

    # set up DARTEL.
    DARTEL = pe.Node(interface=DARTEL, name='DARTEL')
    DARTEL.inputs.image_files = PAG_images
    DARTEL.inputs.iteration_parameters = it_params
    DARTEL.inputs.opt_params = opt_params
    DARTEL.inputs.regularization_form = reg_form

    DARTEL_wf.connect([
        (DARTEL, sinker, [('dartel_flow_fields', 'flow'),
                          ('final_template_file', 'template')])
    ])
