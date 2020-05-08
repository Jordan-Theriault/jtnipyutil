def aCompCor_Muschelli(data_dir, out_dir, subj, task, ventricle_file):
    '''
    data_dir: [string] path to fmriprep output directory, containing subject directories.
    out_dir: [string] path to directory to save aCompCor results to.
    subj: [string] subject ID as it is exists in the fmriprep output directory (e.g. sub-001)
    task: [string] task ID in the BIDS formatted file name.
        e.g. 'stress' for file sub-001_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
        If a task has multiple runs, include the run in this variable.
        e.g. rest_run-01 for sub-001_task-rest_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
    ventricle_file: [string] path and name of ALVIN ventricle mask nifti file.
        Download from https://sites.google.com/site/mrilateralventricle/
    '''
    import os
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from scipy.ndimage.morphology import binary_erosion
    from nilearn.image import resample_img, math_img
    from nilearn.input_data import NiftiMasker
    from sklearn.decomposition import PCA

    print('Performing aCompCor (Muschelli et al. 2014, Neuroimage) on \n subject: %s \n task: %s' % (subj, task))
    if not os.path.isdir(os.path.join(out_dir, subj, 'PCA')):
        os.makedirs(os.path.join(out_dir, subj, 'PCA'))
    if not os.path.isdir(os.path.join(out_dir, subj, 'masks')):
        os.makedirs(os.path.join(out_dir, subj, 'masks'))
    ##### White Matter
    WM_file = nib.load(os.path.join(data_dir, subj, 'anat', subj+'_T1w_space-MNI152NLin2009cAsym_class-WM_probtissue.nii.gz'))
    WM_mask = WM_file.get_fdata()
    WM_mask[WM_mask<(np.max(WM_mask)*.99)] = 0
    WM_mask[WM_mask>=(np.max(WM_mask)*.99)] = 1
    WM_mask = binary_erosion(WM_mask).astype(WM_mask.dtype)
    nib.save(nib.Nifti1Image(WM_mask, WM_file.affine, WM_file.header),
        os.path.join(out_dir, subj, 'masks', subj+'_task-'+task+'_WM-aCompCorMask.nii.gz'))
    ##### Cerebral Spinal Fluid
    CSF_file = nib.load(os.path.join(data_dir, subj, 'anat', subj+'_T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue.nii.gz'))
    CSF_mask = CSF_file.get_fdata()
    CSF_mask[CSF_mask<(np.max(CSF_mask)*.99)] = 0
    CSF_mask[CSF_mask>=(np.max(CSF_mask)*.99)] = 1
    ventricle_mask = resample_img(nib.load(ventricle_file),
                                         target_affine=CSF_file.affine,
                                         target_shape=CSF_file.shape,
                        interpolation='nearest')
    CSF_mask = np.multiply(CSF_mask,ventricle_mask.get_fdata())
    nib.save(nib.Nifti1Image(CSF_mask, CSF_file.affine, CSF_file.header),
        os.path.join(out_dir, subj, 'masks', subj+'_task-'+task+'_CSF-aCompCorMask.nii.gz'))

    ##### Mask functional data and perform PCA.
    func_file = nib.load(os.path.join(data_dir, subj, 'func', subj+'_task-'+task+'_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'))
    conf_file = os.path.join(data_dir, subj, 'func', subj+'_task-'+task+'_bold_confounds.tsv')

    CSF_fit = resample_img(nib.load(os.path.join(out_dir, subj, 'masks', subj+'_task-'+task+'_CSF-aCompCorMask.nii.gz')),
                           target_affine=func_file.affine,
                           target_shape=func_file.shape[0:3],
                           interpolation='nearest')
    WM_fit = resample_img(nib.load(os.path.join(out_dir, subj, 'masks', subj+'_task-'+task+'_WM-aCompCorMask.nii.gz')),
                          target_affine=func_file.affine,
                          target_shape=func_file.shape[0:3],
                          interpolation='nearest')
    # White Matter PCA
    niftimask = NiftiMasker(dtype='float32')
    niftimask.fit(math_img('img > 0', img=WM_fit))
    func_WM = niftimask.transform(func_file)
    runPCA = PCA()
    PCA_WM = runPCA.fit(func_WM)
    WM_out = pd.DataFrame(PCA_WM.components_)
    WM_out.columns = ['aCompCorWMnoFilter' + str(col) for col in WM_out.columns]
    WM_out.to_csv(os.path.join(out_dir, subj, 'PCA', subj+'_task-'+task+'_WM-aCompCor.csv'), index=False)

    # CSF PCA
    niftimask = NiftiMasker(dtype='float32')
    niftimask.fit(math_img('img > 0', img=CSF_fit))
    func_CSF = niftimask.transform(func_file)
    runPCA = PCA()
    PCA_CSF = runPCA.fit(func_CSF)
    CSF_out = pd.DataFrame(PCA_CSF.components_)
    CSF_out.columns = ['aCompCorCSFnoFilter' + str(col) for col in CSF_out.columns]
    CSF_out.to_csv(os.path.join(out_dir, subj, 'PCA', subj+'_task-'+task+'_CSF-aCompCor.csv'), index=False)

    # % Variance components
    var_out = pd.DataFrame({
        'var_WM':PCA_WM.explained_variance_,
        'percent_var_WM':PCA_WM.explained_variance_ratio_,
        'var_CSF':PCA_CSF.explained_variance_,
        'percent_var_CSF':PCA_CSF.explained_variance_ratio_,
        })
    var_out.to_csv(os.path.join(out_dir, subj, 'PCA', subj+'_task-'+task+'_variance-aCompCor.csv'))
