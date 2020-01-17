'''
docker run -it --rm \
-v ~/projects/NCI_U01/FSMAP_data/BIDS_modeled:/home/neuro/data \
-v ~/projects/NCI_U01/FSMAP_data/BIDS_modeled_2lvl:/home/neuro/output \
-v ~/projects/NCI_U01/scripts:/home/neuro/scripts \
-v ~/projects/NCI_U01/FSMAP_data/workdir2:/home/neuro/workdir \
-v ~/projects/NCI_U01/FSMAP_data/BIDS_fmriprep/func:/home/neuro/func \
-v ~/atlases/:/home/neuro/atlases \
-v ~/github:/home/neuro/homeutils \
-p 8888:8888 \
jtheriaultpsych:jtnipyutil

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
'''
import os, glob
import nibabel as nib
from jtnipyutil.util import inflate_volumetric_ROI

roi_dir = '/home/neuro/atlases/Glasser atlas/v4_2009cAsym_uninflated/niftis'
work_dir = '/home/neuro/workdir/inflate_ROIs'
target_roi_list = ['L_a24pr_ROI.nii', 'R_a24pr_ROI.nii']
vox_dilate = 4
mni_gm_file = '/home/neuro/atlases/MNI/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii'
gm_thresh = .05
x_axis_midpoint = 129 # in voxel space.
l_hem = 'lh.L_'
r_hem = 'rh.R_'
kmean_num = 0

# # midline MCC
inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)

# # midline ACC
target_roi_list = ['L_25_ROI.nii', 'L_a24_ROI.nii',
                   'R_25_ROI.nii', 'R_a24_ROI.nii']
x_axis_midpoint = 130 # in voxel space.
inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)

# insula
x_axis_midpoint = 129 # in voxel space.
target_roi_list = ['L_AAIC_ROI.nii', 'L_FOP3_ROI.nii', 'L_Ig_ROI.nii',
                  'R_AAIC_ROI.nii', 'R_FOP3_ROI.nii', 'R_Ig_ROI.nii']
inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)

ant_insula_list = glob.glob(os.path.join(work_dir, '*AAIC_ROI*'))
insula_split_R = 92 # X vox coordinate to split R ventral insula into dorsal/medial.
insula_split_L = 164 # X vox coordinate to split L ventral insula into dorsal/medial.
for file in ant_insula_list:
    if 'rh.R' in file:
        insula_data = nib.load(file).get_data()
        insula_data[insula_split_R:, :, :] = 0
        nib.save(nib.Nifti1Image(insula_data, nib.load(file).affine, nib.load(file).header),
                 os.path.join(work_dir, '.'.join(file.split('/')[-1].split('.')[:-1])+'lateral.nii'))
        insula_data = nib.load(file).get_data()
        insula_data[:insula_split_R, :, :] = 0
        nib.save(nib.Nifti1Image(insula_data, nib.load(file).affine, nib.load(file).header),
                 os.path.join(work_dir, '.'.join(file.split('/')[-1].split('.')[:-1])+'medial.nii'))
    if 'lh.L' in file:
        insula_data = nib.load(file).get_data()
        insula_data[insula_split_L:, :, :] = 0
        nib.save(nib.Nifti1Image(insula_data, nib.load(file).affine, nib.load(file).header),
                 os.path.join(work_dir, '.'.join(file.split('/')[-1].split('.')[:-1])+'medial.nii'))
        insula_data = nib.load(file).get_data()
        insula_data[:insula_split_L, :, :] = 0
        nib.save(nib.Nifti1Image(insula_data, nib.load(file).affine, nib.load(file).header),
                 os.path.join(work_dir, '.'.join(file.split('/')[-1].split('.')[:-1])+'lateral.nii'))

# new more lenient GM threshold.
gm_thresh = .05
# control
target_roi_list = ["L_p9-46v_ROI.nii", "L_7PC_ROI.nii", "L_LO1_ROI.nii",
                  "R_p9-46v_ROI.nii", "R_7PC_ROI.nii", "R_LO1_ROI.nii",]
inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)
# control and cluster
target_roi_list = ["L_V1_ROI.nii", "L_1_ROI.nii",
                  "R_V1_ROI.nii", "R_1_ROI.nii"]
kmean_num = 3
inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)

# sort V1 and M1 clusters.
def sort_files_by_axis(file_list, dim_num):
    '''
    dim_num = 0: y axis
    dim_num = 1: z axis
    '''
    import numpy as np
    import os
    min_list = []
    for file in file_list:
        dim_max = np.amax(nib.load(file).get_data(), axis=0)
        min_list.append([np.min(np.where(dim_max>=1)[dim_num]), file]) # max along y axis.
        min_list = sorted(min_list, key = lambda x: x[0]) # sort then rename from back to front.
    for idx, cluster in enumerate(min_list):
        file_num = idx+1
        os.rename(cluster[1], cluster[1].split('-')[0]+'-'+str(file_num)+'_temp_.nii')
        min_list[idx][1] = cluster[1].split('-')[0]+'-'+str(file_num)+'_temp_.nii'
    for idx, cluster in enumerate(min_list):
        os.rename(cluster[1], ''.join(cluster[1].split('_temp_')[:]))

sort_files_by_axis(glob.glob(os.path.join(work_dir, '*L_V1_ROI_kMeanCluster-*')), 0)
sort_files_by_axis(glob.glob(os.path.join(work_dir, '*R_V1_ROI_kMeanCluster-*')), 0)
sort_files_by_axis(glob.glob(os.path.join(work_dir, '*L_1_ROI_kMeanCluster-*')), 1)
sort_files_by_axis(glob.glob(os.path.join(work_dir, '*R_1_ROI_kMeanCluster-*')), 1)
