def get_cortical_thickness(subj, data_dir, work_dir, space=''):
    import nighres
    import nibabel as nib
    import numpy as np

    if space:
        space = '_space-'+space
    # Load tissue classification, transform into binary mask of white matter.
    segfile = nib.load(data_dir+'/'+subj+'/anat/'+subj+'_T1w'+space+'_dtissue.nii.gz')
    segdata = segfile.get_data()
    segdata[segdata!=3] = 0
    segdata[segdata==3] = 1
    segout = nib.Nifti1Image(segdata, affine = segfile.affine, header = segfile.header)
    nib.save(segout, work_dir+'/'+subj+space+'_dWM.nii.gz')

    # Use cruise to generate levelsets for volumetric_layering.
    cruise_sub = nighres.cortex.cruise_cortex_extraction(
        init_image=work_dir+'/'+subj+space+'_dWM.nii.gz', # binary wm mask
        wm_image=data_dir+'/'+subj+'/anat/'+subj+'_T1w'+space+'_class-WM_probtissue.nii.gz', # probability wm mask
        gm_image=data_dir+'/'+subj+'/anat/'+subj+'_T1w'+space+'_class-GM_probtissue.nii.gz', # probability gm mask
        csf_image=data_dir+'/'+subj+'/anat/'+subj+'_T1w'+space+'_class-CSF_probtissue.nii.gz', # probability csf mask
        normalize_probabilities=True,
        save_data=True,
        file_name=''+subj+space+'_cruise',
        output_dir=work_dir)

    # use volumetric labeling to create cortical depth layering.
    cruise_sub = nighres.laminar.volumetric_layering(
        inner_levelset=cruise_sub['gwb'],
        outer_levelset=cruise_sub['cgb'],
        n_layers=10,
        save_data=True,
        file_name=subj+space+'_depth',
        output_dir=work_dir)

import os
subj = os.environ['SUBJ']
data_dir = '/scratch/'+os.environ['USER']+'/'+os.environ['SUBJ']+'/'+os.environ['PROJNAME']+'/BIDS_fmriprep'
work_dir = '/scratch/'+os.environ['USER']+'/'+os.environ['SUBJ']+'/'+os.environ['PROJNAME']+'/nighres'

get_cortical_thickness(subj, data_dir, work_dir, space=os.environ['SPACE'])
