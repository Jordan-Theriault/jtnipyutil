import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
from scipy.stats.stats import pearsonr
# mpl.use('Agg')
# import matplotlib.pyplot as plt
from nilearn.image import resample_img, math_img
from nilearn.input_data import NiftiMasker

root = '/scratch/'+os.environ['USER']+'/'+os.environ['SUBJ']+'/'+os.environ['PROJNAME']
# sanity_check = True
subj = os.environ['SUBJ']
depth_file = root+'/data/'+os.environ['DEPTH']
img_file = root+'/data/'+os.environ['IMG']
conf_file = root+'/data/'+os.environ['CONFOUND']
work_dir = root+'/nighres'
atlas = os.environ['ATLAS']
# conf_names = ['^AROMAAggrComp', 'WhiteMatter', 'CSF'] # see Ciric et al., 2017, Neuroimage SET VARIABLE TO INCLUDE CONFOUNDS. ACCEPTS A STRING, OR A LIST OF STRINGS.
conf_names = 'TCompCor' # see following ANTS normalization, used in Kashyap et al., 2018, Sci Reports
detrend_tf = False

######## Fit cortical depth to the resting state data (affine and shape)
fit_depth = resample_img(nib.load(depth_file),
                                 target_affine=nib.load(img_file).affine,
                                 target_shape=nib.load(img_file).shape[0:3])
fit_depth_data = np.round(fit_depth.get_data(), 2) # rounding to 2 here kept the distribution similar to when the cortical depth was not transformed to bold space.
nib.save(nib.Nifti1Image(fit_depth_data, fit_depth.affine,
                         fit_depth.header), os.path.join(work_dir, 'ALIGN_'+depth_file.split('/')[-1]))
depth_refit = nib.load(os.path.join(work_dir, 'ALIGN_'+depth_file.split('/')[-1]))

######## Fit atlas to the resting state data (affine and shape)
print('getting atlas data from file: %s' % atlas)
atlas_file = root+'/data/'+atlas
fit_atlas = resample_img(nib.load(atlas_file),
                                 target_affine=nib.load(img_file).affine,
                                 target_shape=nib.load(img_file).shape[0:3],
                interpolation='nearest')
fit_atlas_data = np.round(fit_atlas.get_data(), 0)
nib.save(nib.Nifti1Image(fit_atlas_data, fit_atlas.affine,
                     fit_atlas.header), os.path.join(work_dir, 'ALIGN_'+atlas_file.split('/')[-1]))
atlas_refit = nib.load(os.path.join(work_dir, 'ALIGN_'+atlas_file.split('/')[-1]))

######## Model whole-brain BOLD data using motion confounds.
niftimask = NiftiMasker(dtype='float32', t_r=2.34, detrend=detrend_tf)
niftimask.fit(math_img('img > 0', img=depth_refit))
######## Get Confounds, if specified.
if conf_names:
    conf_raw = pd.read_csv(conf_file, sep='\t')
    if isinstance(conf_names, list):
        confound_list = conf_raw[conf_raw.columns[conf_raw.columns.to_series().str.contains('|'.join(conf_names))]]
    else:
        confound_list = conf_raw[conf_raw.columns[conf_raw.columns.to_series().str.contains(conf_names)]]

img_masked = niftimask.transform(img_file, confounds=confound_list.values)

######## Mask atlas by cortical depth.
niftimask_atlas = NiftiMasker(dtype='float32')
niftimask_atlas.fit(math_img('img > 0', img=depth_refit))
atlas_masked = niftimask_atlas.transform(atlas_refit)

######## Mask cortical depth by cortical depth.
niftimask_depth = NiftiMasker(dtype='float32')
niftimask_depth.fit(math_img('img > 0', img=depth_refit))
depth_masked = niftimask_atlas.transform(depth_refit)

assert img_masked.shape[-1] == atlas_masked.shape[-1], 'flattened image and flattened atlas are not equal on dimension 0'
corr_list = []
roi_output_10 = []
roi_countout_10 = []
roi_output_3 = []
roi_countout_3 = []

######## Calculate correlation (raw depth vs BOLD).
for idx, roi in enumerate(np.unique(atlas_masked)[1:]):
    ######## Correlate cortical depth with BOLD
    roi_data = img_masked[:, [atlas_masked == roi][0][0][:]]
    roi_depth = depth_masked[0][[atlas_masked == roi][0][0][:]]
    roi_vox_avg = np.average(roi_data, axis=0)
    try: corr_list = np.vstack((corr_list,
                        [pearsonr(roi_vox_avg, roi_depth)[0], # Correlation between BOLD and cortical depth.
                         pearsonr(roi_vox_avg, roi_depth)[1], # p value
                         roi_data.shape[-1]]))
    except:
        corr_list = [pearsonr(roi_vox_avg, roi_depth)[0], # Correlation between BOLD and cortical depth.
                         pearsonr(roi_vox_avg, roi_depth)[1], # p value
                         roi_data.shape[-1]]
    ######## Calculate depth (10 categories)
    bins = np.linspace(0,1,11)
    roi_depth_10 = np.digitize(roi_depth, bins, right=True)
    roi_data_10 = [roi_data[:, roi_depth_10 == i].mean(axis=1) for i in range(1, len(bins))] # average across 10 bins
    roi_count_10 = [(roi_depth_10 == i).sum() for i in range(1, len(bins))] # N of voxels in each bin
    try:
        print('loading data from roi %s into 10-category depth output' % roi)
        roi_output_10 = np.vstack((roi_output_10, roi_data_10))
        roi_countout_10 = np.hstack((roi_countout_10, roi_count_10))
    except:
        print('starting 10 depth category output')
        roi_output_10 = roi_data_10
        roi_countout_10 = roi_count_10
    ######## Calculate depth (3 categories)
    bins = np.linspace(0,1,4)
    roi_depth_3 = np.digitize(roi_depth, bins, right=True)
    roi_data_3 = [roi_data[:, roi_depth_3 == i].mean(axis=1) for i in range(1, len(bins))] # average across 3 bins
    roi_count_3 = [(roi_depth_3 == i).sum() for i in range(1, len(bins))] # N of voxels in each bin
    try:
        print('loading data from roi %s into 3-category depth output' % roi)
        roi_output_3 = np.vstack((roi_output_3, roi_data_3))
        roi_countout_3 = np.hstack((roi_countout_3, roi_count_3))
    except:
        print('starting 3 depth category output')
        roi_output_3 = roi_data_3
        roi_countout_3 = roi_count_3

######## Save correlations
corr_out = pd.DataFrame({'subj':np.repeat(subj, len(np.unique(atlas_masked)[1:])),
                         'roi':np.unique(atlas_masked)[1:],
                         'BOLD_depth_corr':corr_list[:,0],
                        'p_val':corr_list[:,1],
                        'ROI_n':corr_list[:,2]})
corr_out.to_csv(os.path.join(work_dir, 'correlation_'+atlas.split('.')[0]+'_'+img_file.split('/')[-1].split('.nii.gz')[0]+'.csv'),
             index=False, header=True)

# ######## Save depth 10
depth10_out = pd.DataFrame({'subj':np.repeat(subj, len(np.repeat(np.unique(atlas_masked)[1:], 10))),
                            'roi':np.repeat(np.unique(atlas_masked)[1:], 10),
                            'depth_cat':list(range(0,10))*len(np.unique(atlas_masked)[1:]),
                            'cat_N':roi_countout_10})
depth10_temp = pd.DataFrame(roi_output_10)
depth10_out = depth10_out.join(depth10_temp)
depth10_out.to_csv(os.path.join(work_dir, 'depth10'+atlas.split('.')[0]+'_'+img_file.split('/')[-1].split('.nii.gz')[0]+'.csv'),
             index=False, header=True)

######## Save depth 3
depth3_out = pd.DataFrame({'subj':np.repeat(subj, len(np.repeat(np.unique(atlas_masked)[1:], 3))),
                            'roi':np.repeat(np.unique(atlas_masked)[1:], 3),
                            'depth_cat':list(range(0,3))*len(np.unique(atlas_masked)[1:]),
                            'cat_N':roi_countout_3})
depth3_temp = pd.DataFrame(roi_output_3)
depth3_out = depth3_out.join(depth3_temp)
depth3_out.to_csv(os.path.join(work_dir, 'depth3'+atlas.split('.')[0]+'_'+img_file.split('/')[-1].split('.nii.gz')[0]+'.csv'),
             index=False, header=True)

########  Replace values in atlas and save nifti.
out_atlas = np.zeros(fit_atlas_data.shape)
for idx, corr in enumerate(corr_list):
    out_atlas[fit_atlas_data==np.unique(atlas_masked)[1:][idx]] = corr[0]

atlas_head = fit_atlas.header
atlas_head['cal_max'] = 1
atlas_head['cal_min'] = 0
nib.save(nib.Nifti1Image(out_atlas, fit_atlas.affine,
                     atlas_head), os.path.join(work_dir, 'CORR_PSC_w_DEPTH_'+subj+'_'+atlas_file.split('/')[-1]))

print('done with %s' % subj)
