def convert_affine(native_img, ref_img, work_dir):
    '''
    Transforms image from native space to a reference space, using nibabel.apply_affine.

    Input [Mandatory]:
    native_img: string, giving file path to image in native space.
    ref_img: string, giving file path to image in reference space.
    work_dir: string, representing directory to save output to.

    Output:
        trans_img: native image in tranformed space.
    '''
    import nibabel as nb
    import nilearn as nl
    import numpy as np
    import os

    native = nib.load(native_img)
    ref = nib.load(ref_img)
    out_img = nl.image.resample_img(native, target_affine=ref.affine, target_shape=ref.shape)
    out_data = out_img.get_data()
    out_data = np.round(out_data, 5)
    out_img = nib.Nifti1Image(out_data, out_img.affine, out_img.header)
    nib.save(out_img, os.path.join(work_dir, 'ALIGN_'+native_img.split('/')[-1]))
    return os.path.join(work_dir, 'ALIGN_'+native_img.split('/')[-1])

def align_mask(mask_file, native_brain, ref_brain, work_dir):
    '''
    Aligns a mask to a reference space, given a native-space brain and reference space brain.

    Input [Mandatory]:
    mask_file: string, giving file path to binary nifti file.
    native_brain: string, giving file path to T1 brain in mask space.
    ref_brain: string, giving file path to T1 brain in reference space.
    work_dir: string, representing directory to save output to.

    Output:
        aligned_mask: mask .nii.gz file aligned to reference space.
    '''
    import nibabel as nb
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os

    # wf = pe.Workflow(name='align_mask')
    fit_brain = pe.Node(fsl.FLIRT(interp='nearestneighbour'),
                        name='fit_brain')
    fit_brain.inputs.in_file = native_brain
    fit_brain.inputs.reference = ref_brain
    fit_brain.inputs.out_matrix_file = os.path.join(work_dir, mask_file.split('/')[-1]+'_matrix')
    fit_brain.run()

    fit_mask = pe.Node(fsl.FLIRT(interp='nearestneighbour'),
                       name='fit_mask')

    fit_mask.inputs.in_file = mask_file
    fit_mask.inputs.reference = ref_brain
    fit_mask.inputs.in_matrix_file = os.path.join(work_dir, mask_file.split('/')[-1]+'_matrix')
    fit_mask.inputs.apply_xfm = True
    fit_mask.inputs.out_file = os.path.join('/'.join(mask_file.split('/')[0:-1]), 'ALIGN_'+mask_file.split('/')[-1])
    fit_mask.run()
    return fit_mask.inputs.out_file


def create_grandmean_img_wf():
    '''
    Creates a workflow, which creates a grand mean image, averaging within each file, then across all images given.

    The workflow takes the following as input to wf.inputs.inputspec

    Input [Mandatory]:
    wf.inputs.inputspec.subject_list: list of subject IDs
        e.g. [sub-001, sub-002]
    wf.inputs.inputspec.template: dictionary to identify all files.
        e.g. preproc_func_template = {'func': '/home/neuro/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'}

    The template can identify a larger set f files, and the subject_list will grab a subset.
        e.g. The template may grab sub-001, sub-002, sub-003 ...
        But if the subject_list only includes sub-001, then only sub-001 will be used.

    This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

    Output:
        grandmean_wf: workflow to create grand mean across all specified images.
    '''

    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function

    ################## Setup workflow.
    grandmean_wf = pe.Workflow(name='make_ref_img')
    inputspec = pe.Node(IdentityInterface(
        fields=['subject_list', 'template']),
                 name='inputspec')
    outputspec = pe.Node(IdentityInterface(
        fields=['mean_img']),
                        name='outputspec')

    ################## get functional files.
    def get_files(subject_list, template):
        import glob
        out_list = []
        for x in glob.glob(list(template.values())[0]):
            if any(subj in x for subj in subject_list):
                out_list.append(x)
        return out_list

    get_imgs = pe.Node(Function(
        input_names=['subject_list', 'template'],
        output_names=['out_list'],
        function=get_files),
                       name='get_imgs')

    ################## average each subject's functional data.
    mean_imgs = pe.MapNode(fsl.maths.MeanImage(),
        iterfield = ['in_file'],
        name='mean_imgs')

    ################## combine all averaged functional images into one 4d array.
    merge_imgs = pe.Node(interface=fsl.Merge(dimension='t'),
                       name='merge_imgs')
#     merge_func.inputs.in_files = # from mean_func
    merge_imgs.inputs.dimension = 't'

    ################## Then average all average functional images.
    grand_mean = pe.Node(fsl.maths.MeanImage(),
        iterfield = ['in_file'],
        name='grand_mean')


    grandmean_wf.connect([(inputspec, get_imgs, [('subject_list', 'subject_list')]),
                    (inputspec, get_imgs, [('template', 'template')]),
                    (get_imgs, mean_imgs, [('out_list', 'in_file')]),
                    (mean_imgs, merge_imgs, [('out_file', 'in_files')]),
                    (merge_imgs, grand_mean, [('merged_file', 'in_file')]),
                    (grand_mean, outputspec, [('out_file', 'mean_img')]),
                    ])
    return grandmean_wf

def fit_mask(mask_file, ref_file, spline = 0, work_dir = '', out_format = 'file' ):
    '''
    Fits a mask file to the space of a reference image, using nearest neighbor classification.
    Assumes that interpolation happens along 3d axes. All additional dimensions are unchanged.
    Input [Mandatory]:
        mask_file: path to a nifti mask file to be refit to reference space.
        ref_file: path to a nifti file in the reference space. Can be 3d or 4d.
            3d space is the reference dimension.
        spline: spline order for interpolation. Default = 0
        work_dir: [default = ''] path to directory to save masked file. Required if out_format = 'file'.
        out_format: [default = 'file'] Options are 'file', or 'array'.
    Output
        out_mask: Either a nifti file, or a np array, depending on out_format.
    '''
    import numpy as np
    import nibabel as nib
    import os
    from skimage.transform import resize
    mask = nib.load(mask_file)
    mask_name = '_'+mask_file.split('/')[-1].split('.')[0]
    ref = nib.load(ref_file)
    if mask.shape[0:3] != ref.shape[0:3]:
        new_shape = list(ref.shape[0:3])
        while len(new_shape) != len(mask.shape): # add extra dimensions, in case ref img is 4d.
            new_shape.append(1)
        data = resize(mask.get_data(), new_shape, order=spline, preserve_range=True) # interpolate mask to native space.
    else:
        print('mask is already in reference space!')
        data = mask.get_data()

    if out_format == 'file':
        if work_dir == '':
            print('No save directory specified, saving to current working direction')
            work_dir = os.getcwd()
        out_mask = nib.Nifti1Image(data, ref.affine, mask.header)
        out_mask.header['dim'] = ref.header['dim']
        out_mask.header['pixdim'] = ref.header['pixdim']
        out_mask.header['cal_max'] = np.max(data) # adjust min and max header info.
        out_mask.header['cal_min'] = np.min(data)
        nib.save(out_mask, os.path.join(work_dir, mask_name + '_spline'+str(spline)+'_fit.nii.gz'))
    else:
        assert (out_format == 'array'), 'out_format is neither file, or array.'
        out_mask = data

    return out_mask

def mask_img(img_file, mask_file, work_dir = '', out_format = 'file', inclu_exclu = 'inclusive', spline = 0):
    '''
    Applies a mask, converting to reference space if necessary, using nearest neighbor classification.
    Input [Mandatory]:
        img_file: path to a nifti file to be masked. Can be 3d or 4d.
        mask_file: path to a nifti mask file. Does not need to match dimensions of img_file
        work_dir: [default = ''] path to directory to save masked file. Required if out_format = 'file'.
        out_format: [default = 'file'] Options are 'file', or 'array'.
        inclu_exclu: [default = 'exclusive'] Options are 'exclusive' and 'inclusive'
    Output
        out_img: Either a nifti file, or a np array, depending on out_format.
    '''
    import numpy as np
    import nibabel as nib
    import os.path
    from skimage.transform import resize
    print(('loading mask file: %s') % mask_file)
    mask = nib.load(mask_file)
    mask_name = '_'+mask_file.split('/')[-1].split('.')[0]
    img_name = img_file.split('/')[-1].split('.')[0]
    print(('loading img file: %s') % img_file)
    img = nib.load(img_file) # grab data
    data = nib.load(img_file).get_data() # grab data
    if mask.shape != data.shape[0:3]:
        new_shape = list(data.shape[0:3])
        while len(new_shape) != len(mask.shape): # add extra dimensions, in case ref img is 4d.
            new_shape.append(1)
        mask = resize(mask.get_data(), new_shape, order = spline, preserve_range=True) # interpolate mask to native space.
    else:
        mask = mask.get_data()
    if inclu_exclu == 'inclusive':
        data[mask!=1] = 0 # mask
    else:
        assert(inclu_exclu == 'exclusive'), 'mask must be either inclusive or exclusive'
        data[mask==1] = 0 # mask

    if out_format == 'file':
        if work_dir == '':
            print('No save directory specified, saving to current working direction')
            work_dir = os.getcwd()
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(out_img, os.path.join(work_dir, img_name + mask_name + '.nii.gz'))
        out_img = os.path.join(work_dir, img_name + mask_name + '.nii.gz')
    else:
        assert (out_format == 'array'), 'out_format is neither file, or array.'
        out_img = data

    return out_img

def clust_thresh(img, thresh=95, cluster_k=50):
    '''
    Thresholds an array, then computes sptial cluster.
    Input [Mandatory]:
        img: 3d array - e.g. from nib.get_data()
    Input:
        thresh: % threshold extent.
            Default = 95
        cluster_k: k-voxel cluster extent. integer or list.
            Default = 50.
            A list can be used to give fallback clusters. e.g. cluster_k= [50, 40]
                In this case, the first threhsold is used,
                and if nothing passes it then we move onto the next.
    Output:
        out_labeled: 3d array, with values 1:N for clusters, and 0 otherwise.
    '''
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    out_labeled = np.empty((img.shape[0], img.shape[1],img.shape[2]))
    data = img[np.where(~np.isnan(img))] # strip out data, to avoid np.nanpercentile.
    data = data[data > 0]
    img = np.nan_to_num(img)
    img[img < np.percentile(data, thresh)] = 0 #threshold residuals.
    label_map, n_labels = label(img) # label remaining voxels.
    lab_val = 1 # this is so that labels are ordered sequentially, rather than having gaps.
    if type(cluster_k) == int:
        print(('looking for clusters > %s voxels, at %s%% threshold') % (cluster_k, thresh))
        for label_ in range(1, n_labels+1): # addition is to match labels, which are base 1.
            if np.sum(label_map==label_) >= cluster_k:
                out_labeled[label_map==label_] = lab_val # zero any clusters below cluster threshold.
                print(('saving cluster %s') % lab_val)
                lab_val = lab_val+1 # add to counter.
    else:
        assert (type(cluster_k) == list), 'cluster_k must either be an integer or list.'
        for k in cluster_k:
            print(('looking for clusters > %s voxels') % k)
            for label_ in range(1, n_labels+1):
                if np.sum(label_map==label_) >= k:
                    out_labeled[label_map==label_] = lab_val
                    print(('saving cluster %s at min %s voxels') % (lab_val, k))
                    lab_val = lab_val+1
            if lab_val > 1: # if we find any clusters above the threshold, then move on. Otherwise, try another threshold.
                break

    return out_labeled


def files_from_template(identity_list, template):
    '''
    Uses glob to grab all matches to a template, then subsets the list with identifier.
    Input [Mandatory]:
        identity_list: string or list of strings, to grab subset of glob search.
            e.g. 'sub-01', or ['sub-01', 'sub-02', 'sub-03']
        template: string denoting a path, with wildcards, to be used in glob.
            e.g. '/home/neuro/data/smoothsub-*/model/sub-*.nii.gz'
    Output:
        out_list: list of file paths, first from the glob template,
        then subseted by identifier.
    '''
    import glob
    out_list = []
    if type(identity_list) != list:
        assert (type(identity_list) == str), 'identifier must be either a string, or a list of string'
        identity_list = [identity_list]
    for x in glob.glob(template):
        if any(subj in x for subj in identity_list):
            out_list.append(x)
    return out_list

def combine_runs(subj, out_folder, runs=False, bold_template = False, bmask_template = False, task_template = False, conf_template = False):
    '''
    Combines bold, task, or confound files in fmriprep folder, ASUMMNGS BIDS FOLDER STRUCTURE.
    Also can create an inclusive mask, keeping only voxels shared across all runs.
    For task, will also create a run_onset column, which lists onsets (in seconds) relative to the start of the run. The normal 'onset' column gives a unique onset value for each event.
    For confounds, ICAAroma components will have the run number appended to the right. So AROMAAggrComp28 becomes AROMAAggrComp28_r1.
    Input [Mandatory]:
        subj: string, denoting subject in BIDS format. e.g. 'sub-03'
        out_folder: string, denoting path to save output to. e.g. '/scratch/wrkdir/beliefphoto'

    Input [Optional, if no templates are given then nothing will happen when the function is run.]
        runs: list of integers, denoting runs to keep. If ommitted, all runs kept.
            e.g. [0, 1, 2, 3]
        bold_template: string, denoting path to all bold files.
            Can (and should) use wildcards. e.g. '/scratch/data/sub-*/func/sub-*_task-beliefphoto_run-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
        bmask_template: string, denoting path to all bold masks.
            Can (and should) use wildcards. e.g. '/scratch/data/sub-*/func/sub-*_task-beliefphoto_run-*_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
        task_template: string, denoting path to all task files.
            Can (and should) use wildcards. e.g. '/scratch/data/sub-*/func/sub-*_task-beliefphoto_run-*_events.tsv'
        conf_template: string, denoting path to all confound files.
            Can (and should) use wildcards. e.g. '/scratch/data/sub-*/func/sub-*_task-beliefphoto_run-*_bold_confounds.tsv'
    '''
    from jtnipyutil.util import mask_img
    import nibabel as nib
    import os
    import pandas as pd
    import numpy as np

    def get_filelist(subj_id, template):
        import glob
        temp_list = []
        out_list = []
        for x in glob.glob(template):
            if subj_id in x:
                out_list.append(x)
        out_list.sort()
        return out_list

    # append all bold data.
    if bold_template:
        if runs:
            file_list = list(get_filelist(subj, bold_template)[i] for i in runs)
        else:
            file_list = get_filelist(subj, bold_template)
        out_file = nib.funcs.concat_images(file_list, axis=3)
        out_boldname = file_list[0].split('/')[-1].partition('run-')[0] + file_list[0].split('/')[-1].partition('run-')[-1][3:]
        try:
            nib.save(out_file, os.path.join(out_folder, out_boldname))
        except:
            os.makedirs(out_folder)
            nib.save(out_file, os.path.join(out_folder, out_boldname))

    # cycle through all masks. Keep only voxels shared in all masks. This mask will be used in the modeling pipeline.
    if bmask_template:
        if runs:
            bmask_list = list(get_filelist(subj, bmask_template)[i] for i in runs)
        else:
            bmask_list = get_filelist(subj, bmask_template)
        for bmask in bmask_list:
            if bmask == get_filelist(subj, bmask_template)[0]:
                out_bmaskname = bmask.split('/')[-1].partition('run-')[0] + bmask.split('/')[-1].partition('run-')[-1][3:]
                bmask_ref = nib.load(bmask)
                fin_bmask = nib.load(bmask).get_data()
                out_bmask = nib.Nifti1Image(fin_bmask, bmask_ref.affine, bmask_ref.header)
                nib.save(out_bmask, os.path.join(out_folder, out_bmaskname)) # save the mask from the first file encountered.
            else:
                fin_bmask = mask_img(os.path.join(out_folder, out_bmaskname), # mask the original mask with each subsequent one.
                                    bmask, out_format='array')
                out_bmask = nib.Nifti1Image(fin_bmask, bmask_ref.affine, bmask_ref.header) # save the new mask.
                nib.save(out_bmask, os.path.join(out_folder, out_bmaskname))

    # append all task data.
    if task_template:
        if runs:
            task_list = list(get_filelist(subj, task_template)[i] for i in runs)
        else:
            print('test')
            task_list = get_filelist(subj, task_template)
        for idx, tfile in enumerate(task_list):
            if tfile == task_list[0]:
                out_taskname = tfile.split('/')[-1].partition('run-')[0] + tfile.split('/')[-1].partition('run-')[-1][3:]
                out_tdata = pd.read_csv(tfile, sep='\t', index_col=None) # start the dataframe if first file.
            else:
                run_tdata = pd.read_csv(tfile, sep='\t', index_col=None)
                out_tdata = out_tdata.append(run_tdata, ignore_index = True)
        for run in out_tdata['run'].unique():
            trial_names = out_tdata['trial_type'].loc[out_tdata['run'] == run].copy()
            trial_names = trial_names + '_r' + run.astype(str)
            out_tdata['trial_type'].loc[out_tdata['run'] == run] = trial_names
        out_tdata.to_csv(os.path.join(out_folder, out_taskname), sep='\t', index=False)

    # append all confound data.
    if conf_template:
        if runs:
            conf_list = list(get_filelist(subj, conf_template)[i] for i in runs)
        else:
            conf_list = get_filelist(subj, conf_template)
        for idx, cfile in enumerate(conf_list):
            if cfile == conf_list[0]:
                out_cdata = pd.read_csv(cfile, sep='\t', index_col=None)
                for col in out_cdata.columns:
                        out_cdata = out_cdata.rename(columns={col: col+'_r'+str(idx+1)})
                out_cdata['run_'+str(idx+1)] = 1
            else:
                run_cdata = pd.read_csv(cfile, sep='\t', index_col=None)
                for col in run_cdata.columns:
                        run_cdata = run_cdata.rename(columns={col: col+'_r'+str(idx+1)})
                run_cdata['run_'+str(idx+1)] = 1
                out_cdata = out_cdata.append(run_cdata, ignore_index = True, sort=False)
        out_cdata[out_cdata.isna()] = 0
        out_confname = cfile.split('/')[-1].partition('run-')[0] + cfile.split('/')[-1].partition('run-')[-1][3:]
        out_cdata.to_csv(os.path.join(out_folder, out_confname), sep='\t', index=False)

def inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                           mni_gm_file='', gm_thresh=1,
                           x_axis_midpoint = '', l_hem = '', r_hem = '', kmean_num = 0):
    '''
    This function iterates through 3d .nii ROI files within a folder and inflates each
    by a specified amount, saving the output

    Input [Mandatory]:
    roi_dir = string referencing a folder with 3d ROI files, each with .nii extension.
        e.g. '/home/neuro/atlases/atlas atlas/v4_2009cAsym_uninflated/niftis'
    work_dir = string to working directory folder. output saved here.
    target_roi_list = list of strings, each with a unique identifier for the ROI.
        e.g. ['L_25_ROI.nii', 'R_25_ROI.nii'].
    vox_dilate = integer, denoting how many voxels to dilate each ROI.

    Input [Optional]:
    mni_gm_file = path to MNI GM probabalistic map .nii file.
        e.g. '/home/neuro/atlases/MNI/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii'
    gm_thresh = numeric, denoting threshold for MNInii'
        e.g. set threshold of .5 for regions near the midline and insula.
        e.g. set threshold to .05 for regions near the cortical surface.
    x_axis_midpoint = integer, denoting x axis midpoint in voxel space.
        If hemispheres are identifed then all voxels to the left/right of this point will be masked.
        This is to avoid inflating midline ROIs into the opposite hemisphere.
    l_hem = string, unique identifier within ROI filenames identifying ROI as in the LEFT hemisphere.
        e.g. 'lh.L_'
    r_hem = string, unique identifier within ROI filenames identifying ROI as in the RIGHT hemisphere.
        e.g. 'rh.R_'
    kmean_num = integer [default = 0], denoting number of k means clusters to create in each ROI.

    Example Input:
    roi_dir = '/home/neuro/atlases/Glasser atlas/v4_2009cAsym_uninflated/niftis'
    work_dir = '/home/neuro/workdir/inflate_ROIs'
    target_roi_list = ['L_25_ROI.nii', 'R_25_ROI.nii']
    vox_dilate = 3
    mni_gm_file = '/home/neuro/atlases/MNI/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii'
    gm_thresh = .5
    x_axis_midpoint = 129 # in voxel space.
    l_hem = 'lh.L_'
    r_hem = 'rh.R_'
    kmean_num = 3

    inflate_volumetric_ROI(roi_dir, work_dir, target_roi_list, vox_dilate,
                       mni_gm_file, gm_thresh, x_axis_midpoint, l_hem, r_hem, kmean_num)
    '''
    import nibabel as nib
    import os
    import glob
    import numpy as np
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import ball
    from nilearn.image import resample_img
    from scipy.ndimage import label
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans

    atlas_rois = glob.glob(os.path.join(roi_dir, '*.nii'))
    if mni_gm_file:
        # Grab gm mask
        MNIgm = nib.load(mni_gm_file)
        # Grab > .5
        MNI_out = nib.Nifti1Image(np.where(MNIgm.get_data()>gm_thresh, 1, 0), MNIgm.affine, MNIgm.header)
        # Resize to atlas space.
        MNI_resize = resample_img(MNI_out,
                                  target_affine=nib.load(atlas_rois[0]).affine,
                                  target_shape=nib.load(atlas_rois[0]).shape[0:3],
                        interpolation='nearest')
        # nib.save(MNI_resize, os.path.join(work_dir, 'MNIgm_resized_binarized.nii.gz'))
    # Grab target ROIs
    for roi in atlas_rois:
        for t_roi in target_roi_list:
            if t_roi in roi:
                print(('ROI: %s') % roi)
                roi_data = nib.load(roi).get_data()
                # dilate
                roi_data = binary_dilation(roi_data, ball(vox_dilate)).astype(roi_data.dtype)
                if mni_gm_file: # chop the data using thresholded MNI, and see if any clusters emerge.
                    roi_data_clust = np.copy(roi_data)
                    roi_data_clust[MNI_resize.get_data() == 0] = 0
                    label_map, n_labels = label(roi_data_clust)
                    if n_labels > 1:
                        # If there are clusters, then grab the largest cluster and return it.
                        unique, counts = np.unique(roi_data_clust, return_counts=True)
                        print(('ROI: %s, \nGrabbing cluster: %s') % (roi, np.argmax(unique[1:])+1))
                        roi_data = np.where(roi_data_clust==(np.argmax(unique[1:])+1), 1, 0)
                if r_hem in roi:
                    roi_data[x_axis_midpoint:,:,:] = 0
                if l_hem in roi:
                    roi_data[:x_axis_midpoint,:,:] = 0
                if kmean_num > 0:# kmeans clustering.
                    roi_xyz = np.where(roi_data == 1) # grab coordinates within mask.
                    roi_xyz_2d = np.array([[x, y, z] for x, y, z in zip(roi_xyz[0], roi_xyz[1], roi_xyz[2])]) # transform 3d coordinates into a 2d array
                    roi_kmeans = KMeans(n_clusters = kmean_num).fit(roi_xyz_2d) # use kmeans clustering on x,y,z coordinates. We are clustering based on distance, s indexed by voxel dimensions.
                    for lab in np.unique(roi_kmeans.labels_): # Now, we fill back in the kmeans labels into the original mask.
                        label_xyz = roi_xyz_2d[roi_kmeans.labels_ == lab]
                        x_coord = [label_xyz[:][coord][0] for coord in range(len(label_xyz))]
                        y_coord = [label_xyz[:][coord][1] for coord in range(len(label_xyz))]
                        z_coord = [label_xyz[:][coord][2] for coord in range(len(label_xyz))]
                        roi_data[x_coord, y_coord, z_coord] = lab+1
                # save and output data.
                roi_data_out = nib.Nifti1Image(roi_data, nib.load(roi).affine, nib.load(roi).header)
                prefix = 'infl'+str(vox_dilate)+'.'+roi.split('/')[-1].split('.nii')[0]
                if gm_thresh:
                    prefix =  'gmthresh'+str(gm_thresh).split('.')[-1]+'.' + prefix
                if kmean_num > 0:
                    prefix = prefix + '_kMeanCluster'
                    for k_out in range(kmean_num): # save separate kmean clusters.
                        roi_data_k = np.copy(roi_data)
                        roi_data_k[roi_data != k_out+1] = 0
                        roi_data_k[roi_data == k_out+1] = 1
                        roi_kdata_out = nib.Nifti1Image(roi_data_k, nib.load(roi).affine, nib.load(roi).header)
                        nib.save(roi_kdata_out, os.path.join(work_dir, prefix + '-' + str(k_out+1)+'.nii'))
                # save full ROI.
                nib.save(roi_data_out, os.path.join(work_dir, prefix + ".nii"))

def kmean_ROI(roi_dir, work_dir, target_roi_list, kmean_num, thresh=1.):
    '''
    This function iterates through 3d .nii ROI files within a folder and performs kmeans clustering on each.

    Input [Mandatory]:
    roi_dir = string referencing a folder with 3d ROI files, each with .nii extension.
        e.g. '/home/neuro/atlases/atlas atlas/v4_2009cAsym_uninflated/niftis'
    work_dir = string to working directory folder. output saved here.
    target_roi_list = list of strings, each with a unique identifier for the ROI.
        e.g. ['L_25_ROI.nii', 'R_25_ROI.nii'].
    kmean_num = integer or list of integers.
        Denotes number of k means clusters to create in all ROIs (if integer), or in each ROI (if a list).
    thresh = real number, referencing cutoff in mask. [Default = 1.]
        This is useful in the case of a probabalistic mask, in which case you might want to set a specfic threshold.

    Example Input:
    roi_dir = '/home/neuro/atlases/Glasser atlas/v4_2009cAsym_uninflated/niftis'
    work_dir = '/home/neuro/workdir/inflate_ROIs'
    target_roi_list = ['L_25_ROI.nii', 'R_25_ROI.nii']
    kmean_num = 3
    OR kmean_num = [3, 5]

    kmean_ROI(roi_dir, work_dir, target_roi_list, kmean_num)
    '''
    import nibabel as nib
    import os
    import glob
    import numpy as np
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans

    atlas_rois = glob.glob(os.path.join(roi_dir, '*.nii'))
    for roi in atlas_rois:
        for idx, t_roi in enumerate(target_roi_list):
            if t_roi in roi:
                if type(kmean_num) == list:
                    kmean_target = kmean_num[idx]
                else:
                    kmean_target = kmean_num
                print(('ROI: %s') % roi)
                roi_data = nib.load(roi).get_data()
                roi_data[np.where(roi_data < thresh)] = 0 # zero everything below the threshold
                roi_xyz = np.where(roi_data >= thresh) # grab coordinates within mask.
                roi_xyz_2d = np.array([[x, y, z] for x, y, z in zip(roi_xyz[0], roi_xyz[1], roi_xyz[2])]) # transform 3d coordinates into a 2d array
                roi_kmeans = KMeans(n_clusters = kmean_target).fit(roi_xyz_2d) # use kmeans clustering on x,y,z coordinates. We are clustering based on distance, s indexed by voxel dimensions.
                for lab in np.unique(roi_kmeans.labels_): # Now, we fill back in the kmeans labels into the original mask.
                    label_xyz = roi_xyz_2d[roi_kmeans.labels_ == lab]
                    x_coord = [label_xyz[:][coord][0] for coord in range(len(label_xyz))]
                    y_coord = [label_xyz[:][coord][1] for coord in range(len(label_xyz))]
                    z_coord = [label_xyz[:][coord][2] for coord in range(len(label_xyz))]
                    roi_data[x_coord, y_coord, z_coord] = lab+1
                # save and output data.
                roi_data_out = nib.Nifti1Image(roi_data, nib.load(roi).affine, nib.load(roi).header)
                prefix = roi.split('/')[-1].split('.nii')[0]+'_kMeanCluster-'+str(kmean_target)
                for k_out in range(kmean_target): # save separate kmean clusters.
                    roi_data_k = np.copy(roi_data)
                    roi_data_k[roi_data != k_out+1] = 0
                    roi_data_k[roi_data == k_out+1] = 1
                    roi_kdata_out = nib.Nifti1Image(roi_data_k, nib.load(roi).affine, nib.load(roi).header)
                    nib.save(roi_kdata_out, os.path.join(work_dir, prefix + '-' + str(k_out+1)+'.nii'))
                # save full ROI.
                nib.save(roi_data_out, os.path.join(work_dir, prefix + ".nii"))

def BIDS_to_dm(F, sampling_freq, run_length, trial_col = 'trial_type', parametric_cols=None, sort=False, keep_separate=True, add_poly=None, unique_cols=[], fill_na=None, **kwargs):
    """
        **
        Modified from nltools.file_reader.onsets_to_dm to accomodate BIDS files,
        customize naming of the trial_type column, and allow parametric modulators.
        **
    This function can assist in reading in one or several BIDS-formated events files, specified in seconds and converting it to a Design Matrix organized as samples X Stimulus Classes.
    Onsets files **must** be organized with columns in the following format:
        1) 'onset, duration, trial_type'

    This can handle multiple runs being given at once (if F is a list), and by default uses separate contrasts for each run.

    Args:
        F (filepath/DataFrame/list): path to file, pandas dataframe, or list of files or pandas dataframes
        TR (float): TR of run.
        run_length (int): number of TRs in the run these onsets came from
        trial_col (string): which column should be used to specify stimuli/trials?
        parametric_cols (list of lists of strings):
        e.g. [['condition1', 'parametric1', 'no_cent', 'no_norm'],
             ['condition2', 'paramatric2', 'cent', 'norm']]
             in each entry:
                 entry 1 is a condition within the trial_col
                 entry 2 is a column in the events folder referenced by F.
                 entry 3 is either 'no_cent', or 'cent', indicating whether to center the parametric variable.
                 entry 4 is either 'no_norm', or 'norm', indicating whether to normalize the parametric variable.
             The condition column specified by entry 1 will be multiplied by the
             parametric weighting specified by entry 2, scaled/centered as specified, then
            appended to the design matrix.
        sort (bool, optional): whether to sort the columns of the resulting
                                design matrix alphabetically; defaults to
                                False
        keep_separate (bool): whether to seperate polynomial columns if reading a list of files and using the addpoly option
        addpoly (int, optional: what order polynomial terms to add as new columns (e.g. 0 for intercept, 1 for linear trend and intercept, etc); defaults to None
        unique_cols (list): additional columns to keep seperate across files (e.g. spikes)
        fill_nam (str/int/float): what value fill NaNs in with if reading in a list of files
        kwargs: additional inputs to pandas.read_csv
    Returns:
        Design_Matrix class
    """
    import pandas as pd
    import numpy as np
    import six
    from nltools.data import Design_Matrix
    from sklearn.preprocessing import scale
    import warnings

    if not isinstance(F, list):
        F = [F]
    out = []
    sampling_freq = 1/TR

    for f in F: ## Loading event files.
        if isinstance(f, six.string_types): # load if file.
            if f.split('.')[-1] == 'tsv':
                df = pd.read_csv(f, **kwargs, sep = '\t') # if .tsv, load with tab separation.
            else:
                df = pd.read_csv(f, **kwargs) # TODO, replace in final code.
        elif isinstance(f, pd.core.frame.DataFrame): #copy if dataframe.
            df = f.copy()
        else:
            raise TypeError("Input needs to be file path or pandas dataframe!")
        # Set onset to closest prior TR.
        df['onset'] = df['onset'].apply(lambda x: int(np.floor(x/TR)))
        ### Build dummy codes for trial column
        X = Design_Matrix(np.zeros([run_length,
                                    len(df[trial_col].unique())]),
                          columns=df[trial_col].unique(),
                          sampling_freq=sampling_freq)
        for i, row in df.iterrows(): # for each entry in the .tsv file, mark a contrast for the duration in the design matrix.
            dur = np.ceil(row['duration']/TR) # round duration to ceiling.
            X.loc[row['onset']-1:row['onset']+dur-1, row[trial_col]] = 1
        if sort:
            X = X.reindex(sorted(X.columns), axis=1) # sort columns.
        ## Parametric modulation, if necessary.
        if parametric_cols:
            par_names = [var[0]+'_'+var[1] for var in parametric_cols] # combine parametric_col indicators to generate new column names.
            XP = Design_Matrix(np.zeros([run_length,
                                         len(par_names)]),
                               columns=par_names,
                               sampling_freq=sampling_freq)
            for idx, cond_par in enumerate(parametric_cols):
                cond = cond_par[0] # get condition to parametrically modulate
                par = cond_par[1] # get name of parametric modulator
                print('modulating conditon', cond, 'by parametric modulator', par)
                if cond_par[2] == 'cent':
                    with_mean=True
                elif cond_par[2] == 'no_cent':
                    with_mean=False
                if cond_par[3] == 'norm':
                    with_std = True
                elif cond_par[3] == 'no_norm':
                    with_std = False
                df[par_names[idx]] = scale(df[par], with_mean=with_mean, with_std=with_std) # scale/center the parametric modulatory
                for i, row in df.iterrows():
                    if row[trial_col] == cond:
                        dur = np.ceil(row['duration']/TR) # round duration to ceiling.
                        if np.isnan(row[par]): # check for missing data.
                            print('NaN found in parameter', par, 'at onset:', row['onset'])
                            XP.loc[row['onset']-1:row['onset']+dur-1] = 0 # remove all data within missing area
                        else:
                            XP.loc[row['onset']-1:row['onset']+dur-1, par_names[idx]] = 1*row[par_names[idx]] # multiple dummy code by parametric modulator.
            X = Design_Matrix(pd.concat([X, XP], axis=1), sampling_freq=sampling_freq) # join parametrc variables to the design.
            out.append(X) # append to other runs, if multiple runs.
    if len(out) > 1:
        out_dm = out[0].append(out[1:], keep_separate=keep_separate, add_poly=add_poly, unique_cols=unique_cols, fill_na=fill_na)
    else:
        if add_poly is not None:
            out_dm = out[0].add_poly(add_poly)
        else:
            out_dm = out[0]
    return out_dm

def get_subj_info(task_file, design_col, confounds, conditions):
    '''
    Makes a Bunch, giving all necessary data about conditions, onsets, and durations to
        FSL first level model. Needs a task file to run.
    Originally used in model.create_lvl1pipe_wf pipeline, but copied here.

    Inputs:
        task file [string], path to the subject events.tsv file, as per BIDS format.
        design_col [string], column name within task file, identifying event conditions to model.
        confounds [pandas dataframe], pd.df of confounds, gathered from get_confounds node.
        conditions [list],
            e.g. ['condition1',
                  'condition2',
                 ['condition1', 'parametric1', 'no_cent', 'no_norm'],
                 ['condition2', 'paramatric2', 'cent', 'norm']]
                 each string entry (e.g. 'condition1') specifies a event condition in the design_col column.
                 each list entry includes 4 strings:
                     entry 1 is a condition within the design_col column
                     entry 2 is a column in the events folder, which will be used for parametric weightings.
                     entry 3 is either 'no_cent', or 'cent', indicating whether to center the parametric variable.
                     entry 4 is either 'no_norm', or 'norm', indicating whether to normalize the parametric variable.
             Onsets and durations will be taken from corresponding values for entry 1
             parametric weighting specified by entry 2, scaled/centered as specified, then
            appended to the design matrix.
    '''
    from nipype.interfaces.base import Bunch
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import scale

    onsets = []
    durations = []
    amplitudes = []
    df = pd.read_csv(task_file, sep='\t', parse_dates=False)
    for idx, cond in enumerate(conditions):
        if isinstance(cond, list):
            if cond[2] == 'no_cent': # determine whether to center/scale
                c = False
            elif cond[2] == 'cent':
                c = True
            if cond[3] == 'no_norm':
                n = False
            elif cond[3] == 'norm':
                n = True
            # grab parametric terms.
            onsets.append(list(df[df[design_col] == cond[0]].onset))
            durations.append(list(df[df[design_col] == cond[0]].duration))
            amp_temp = list(scale(df[df[design_col] == cond[0]][cond[1]].tolist(),
                               with_mean=c, with_std=n)) # scale
            amp_temp = pd.Series(amp_temp, dtype=object).fillna(0).tolist() # fill na
            amplitudes.append(amp_temp) # append
            conditions[idx] = cond[0]+'_'+cond[1] # combine condition/parametric names and replace.
        elif isinstance(cond, str):
            onsets.append(list(df[df[design_col] == cond].onset))
            durations.append(list(df[df[design_col] == cond].duration))
            # dummy code 1's for non-parametric conditions.
            amplitudes.append(list(np.repeat(1, len(df[df[design_col] == cond].onset))))
        else:
            print('cannot identify condition:', cond)
    #             return None
    output = Bunch(conditions= conditions,
                       onsets=onsets,
                       durations=durations,
                       amplitudes=amplitudes,
                       tmod=None,
                       pmod=None,
                       regressor_names=confounds.columns.values,
                       regressors=confounds.T.values.tolist()) # movement regressors added here. List of lists.
    return output

def get_confounds(confound_file, noise_transforms, noise_regressors, TR, options):
    '''
    Gathers confounds (and transformations) into a pandas dataframe.
    Originally used in model.create_lvl1pipe_wf pipeline, but copied here.

    Input [Mandatory]:
        confound_file [string]: path to confound.tsv file, given by fmriprep.
        noise_transforms [list of strings]:
            noise transforms to be applied to select noise_regressors above. Possible values are 'quad', 'tderiv', and 'quadtderiv', standing for quadratic function of value, temporal derivative of value, and quadratic function of temporal derivative.
            e.g. model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
        noise_regressors [list of strings]:
            column names in confounds.tsv, specifying desired noise regressors for model.
            IF noise_transforms are to be applied to a regressor, add '*' to the name.
            e.g. model_wf.inputs.inputspec.noise_regressors = ['CSF', 'WhiteMatter', 'GlobalSignal', 'X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*']
        TR [float]:
            Scanner TR value in seconds.
        options: dictionary with the following entries
            remove_steadystateoutlier [boolean]:
                Should always be True. Remove steady state outliers from bold timecourse, specified in fmriprep confounds file.
            ICA_AROMA [boolean]:
                Use AROMA error components, from fmriprep confounds file.
            poly_trend [integer. Use None to skip]:
                If given, polynomial trends will be added to run confounds, up to the order of the integer
                e.g. "0", gives an intercept, "1" gives intercept + linear trend,
                "2" gives intercept + linear trend + quadratic.
            dct_basis [integer. Use None to skip]:
                If given, adds a discrete cosine transform, with a length (in seconds) of the interger specified.
                    Adds unit scaled cosine basis functions to Design_Matrix columns,
                    based on spm-style discrete cosine transform for use in
                    high-pass filtering. Does not add intercept/constant.
    '''
    import numpy as np
    import pandas as pd
    from nltools.data import Design_Matrix

    df_cf = pd.DataFrame(pd.read_csv(confound_file, sep='\t', parse_dates=False))
    transfrm_list = []
    for idx, entry in enumerate(noise_regressors): # get entries marked with *, indicating they should be transformed.
        if '*' in entry:
            transfrm_list.append(entry.replace('*', '')) # add entry to transformation list if it has *.
            noise_regressors[idx] = entry.replace('*', '')

    confounds = df_cf[noise_regressors]
    transfrmd_cnfds = df_cf[transfrm_list] # for transforms
    TR_time = pd.Series(np.arange(0.0, TR*transfrmd_cnfds.shape[0], TR)) # time series for derivatives.
    if 'quad' in noise_transforms:
        quad = np.square(transfrmd_cnfds)
        confounds = confounds.join(quad, rsuffix='_quad')
    if 'tderiv' in noise_transforms:
        tderiv = pd.DataFrame(pd.Series(np.gradient(transfrmd_cnfds[col]), TR_time)
                              for col in transfrmd_cnfds).T
        tderiv.columns = transfrmd_cnfds.columns
        tderiv.index = confounds.index
        confounds = confounds.join(tderiv, rsuffix='_tderiv')
    if 'quadtderiv' in noise_transforms:
        quadtderiv = np.square(tderiv)
        confounds = confounds.join(quadtderiv, rsuffix='_quadtderiv')
    if options['remove_steadystateoutlier']:
        if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]].empty:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^non_steady_state_outlier')]])
        elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]].empty:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]]) # old syntax
    if options['ICA_AROMA']:
        if not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]].empty:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^aroma_motion')]])
        elif not df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]].empty:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]]) # old syntax
    confounds = Design_Matrix(confounds, sampling_freq=1/TR)
    if isinstance(options['poly_trend'], int):
        confounds = confounds.add_poly(order = options['poly_trend'])
    if isinstance(options['dct_basis'], int):
        confounds = confounds.add_dct_basis(duration=options['dct_basis'])
    return confounds
