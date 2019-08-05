import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nilearn.image import resample_img
from nilearn.input_data import NiftiMasker

root = '/scratch/'+os.environ['USER']+'/'+os.environ['SUBJ']+'/'+os.environ['PROJNAME']
sanity_check = True
subj = os.environ['SUBJ']
depth_file = root+'/data/'+os.environ['DEPTH']
img_file = root+'/data/'+os.environ['IMG']
conf_file = root+'/data/'+os.environ['CONFOUND']

work_dir = root+'/nighres'
mask_list = ['dACC_Wager.rh.sm08.vol.nii.gz',
             'dAmy_Gianaros.rh.sm08.vol.nii.gz',
             'dmIns_Kurth.rh.sm08.vol.nii.gz',
             'dpIns_Gianaros.rh.sm08.vol.nii.gz',
             'lvAIns_Wager.rh.sm08.vol.nii.gz',
             'mvAIns_Harper.rh.sm08.vol.nii.gz',
             'pgACC_Gianaros.rh.sm08.vol.nii.gz',
             'sgACC_Gianaros.rh.sm08.vol.nii.gz'] # TODO this mask list will have to come from the atlas instead.


######## Fit cortical depth to the resting state data (affine and shape), then flatten.
fit_depth = resample_img(nib.load(depth_file),
                                 target_affine=nib.load(img_file).affine,
                                 target_shape=nib.load(img_file).shape[0:3])
fit_depth_data = np.round(fit_depth.get_data(), 2)
# rounding to 2 here kept the distribution similar to when the cortical depth was not transformed to
# bold space.
nib.save(nib.Nifti1Image(fit_depth_data, fit_depth.affine,
                         fit_depth.header), os.path.join(work_dir, 'ALIGN_'+depth_file.split('/')[-1]))
depth_flat = np.reshape(fit_depth_data, np.prod(fit_depth_data.shape))

######## Get AROMA Confounds
conf_raw = pd.read_csv(conf_file, sep='\t')
conf_AROMA = conf_raw[conf_raw.columns[conf_raw.columns.to_series().str.contains('^AROMAAggrComp')]]
fig, ax = plt.subplots(nrows=len(mask_list), ncols=2)

for idx, mask in enumerate(mask_list):
    print('getting data for mask: %s' % mask.split('.')[0])
    # Fit mask to the resting state data (affine and shape).
    mask_file = root+'/data/'+mask
    fit_mask = resample_img(nib.load(mask_file),
                                     target_affine=nib.load(img_file).affine,
                                     target_shape=nib.load(img_file).shape[0:3],
                    interpolation='nearest')
    fit_mask_data = np.round(fit_mask.get_data(), 0)
    nib.save(nib.Nifti1Image(fit_mask_data, fit_mask.affine,
                         fit_mask.header), os.path.join(work_dir, 'ALIGN_'+mask_file.split('/')[-1]))

    # Mask flattened depth
    mask_flat = np.reshape(fit_mask_data, np.prod(fit_mask_data.shape))
    depth_masked = depth_flat[mask_flat>0]

    # Mask BOLD data, modeling motion confounds.
    niftimask = NiftiMasker(dtype='float32', t_r=2.34)
    niftimask.fit(fit_mask)
    img_masked = niftimask.transform(img_file, confounds=conf_AROMA.values)
    assert img_masked.shape[1] == len(depth_masked), 'masked image and masked depth are not equal length'

    # remove any additional voxels with no cortical depth
    img_masked = img_masked[:, depth_masked != 0]
    depth_masked = depth_masked[depth_masked != 0]

    #  Get cortical depth in Native space for histogram.
    depthfit_mask = resample_img(nib.load(mask_file),
                                 target_affine=nib.load(depth_file).affine,
                                 target_shape=nib.load(depth_file).shape[0:3])
    depthfit_mask_data = np.round(depthfit_mask.get_data(), 0)
    depth_native = nib.load(depth_file).get_data()[depthfit_mask_data>0]
    depth_native = depth_native[depth_native>0]

    # plot and save histogram of cortical depth (TRANSFORMED & NATIVE SPACE)
    ax[idx, 0].hist(depth_masked, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    ax[idx, 0].set_title('ROI = '+ mask.split('.')[0]+', Transformed')
#     plt.ylabel('COUNT')
#     plt.xlabel('CORTICAL DEPTH')
    ax[idx, 1].set_title('Native')
    ax[idx, 1].hist(depth_native, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])

    # Sanity check, to make sure mask fits properly
    if sanity_check:
        fit_depth_data_temp = np.empty_like(fit_depth_data)
        fit_depth_data_temp[fit_mask_data==1] = fit_depth_data[fit_mask_data==1]
        fit_depth_data_temp[fit_mask_data!=1] = 0
        nib.save(nib.Nifti1Image(fit_depth_data_temp, fit_depth.affine,
                             fit_depth.header), os.path.join(work_dir, 'sanitycheck_'+mask.split('.')[0]+'_'+depth_file.split('/')[-1]))

    # Then save the TRs as an array, with the depth as the first row.
    depth_df = pd.DataFrame(depth_masked).transpose()
    out_df = depth_df.append(pd.DataFrame(img_masked), ignore_index=True)
    out_df.to_csv(os.path.join(work_dir, 'depth_and_data_'+mask.split('.')[0]+'_'+img_file.split('/')[-1].split('.nii.gz')[0]+'.csv'), # TODO rather than saving this, we'll need to calculate the correlation (signal ~ depth) and save it for the subject and ROI. Output should be a long format table of ROI, subjects, and correlation values. Then, calculate the average correlation (and SE) for each ROI, and plot on an atlas.
                 index=False, header=False)

plt.tight_layout()
plt.savefig(work_dir + '/corticaldepth_histogram_'+subj+'.png')
