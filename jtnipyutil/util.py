
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
        nib.save(out_mask, os.path.join(work_dir, mask_name + '_spline'+str(spline)+'_fit.nii.gz'))
        out_mask.header['cal_max'] = np.max(data) # adjust min and max header info.
        out_mask.header['cal_min'] = np.min(data)
        out_mask = os.path.join(work_dir, mask_name + '_spline'+str(spline)+'_fit.nii.gz')
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
        interp_dims = np.array(data.shape[0:3])/np.array(mask.shape)
        mask = resize(mask.get_data(), interp_dims.tolist(), order = spline, preserve_range=True) # interpolate mask to native space.
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

def sphere_roi_from_list(template_file, subj_list, coord_list, radius, out_dir, out_name, roi_value=1):
    '''
    Given a set of subjects and coordinates, iterate through all entries and generate ROI spheres around the coordinates.
    Value of ROI can be customized so that entries can later be combined into a single atlas.

    template_file [string]:
        path to 3-dimensional coordinate space file. Used to generate image in the correct voxel space.
        e.g. ~/Desktop/data/TRAG/TRAG/ROIs/ToMN/images/mni_icbm152_t1_tal_nlin_sym_09c.nii
    subj_list [list of strings]:
        Each entry should uniquely identify a subject (e.g. ['sub-03', 'sub-04'])
    coord_list [list of lists of integers]:
        e.g. [[45, 74, 51], [96, 72, 103]]
    radius [integer]:
        radius of sphere
    wrk_dir [string]:
        path to output directory
    out_name [string]
        identifier for output file, e.g. 'RTPJ'
    roi_value [integer]:
        value to use inside sphere. Default=1
        Can be useful to change this, if you plan on combining ROIs later.
    '''
    import nibabel as nib
    import numpy as np
    import os
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import ball

    template_img = nib.load(template_file)
    data = template_img.get_data()
    for idx, subj in enumerate(subj_list):
        subj_coords = coord_list[idx]
        if np.isnan(subj_coords).any(): # if any nan found, then skip this subject.
            print(('invalid coordinates for subject %s') % subj)
            continue
        data[:] = 0
        data[int(coord_list[idx][0]), int(coord_list[idx][1]), int(coord_list[idx][2])] = 1
        data = binary_dilation(data, ball(radius)).astype(data.dtype)
        data = data*roi_value
        out_file = nib.Nifti1Image(data, template_img.affine, template_img.header)
        out_file.header['cal_max'] = roi_value # fix header info
        out_file.header['cal_min'] = 0 # fix header info
        try:
            nib.save(out_file, os.path.join(out_dir, subj+'_'+out_name+'ROI_sphere_'+str(radius)+'mm.nii'))
        except:
            os.makedirs(out_dir)
            nib.save(out_file, os.path.join(out_dir, subj+'_'+out_name+'ROI_sphere_'+str(radius)+'mm.nii'))

def combine_masks(dir_template, subj_list, out_dir, out_name, mask_template = False):
    '''
    Combine multiple ROI masks into a single atlas.
    ROIs should each have separate ID values before being entered
    TODO - allow all masks to be binary, then label in script, and output a legend.

    dir_template [string]:
        file path that will grab all relevant files with a glob. subject IDs will be used to find files for each subject.
    subj_list [list of strings]:
        these will be iterated through to identify each subject. If we are combining data for just one subject (or all subjects) then the entry will just need to be present in all filenames.
    out_dir [string]:
        path to output directory. Directory will be made if it does not already exist.
    out_name [string]:
        Unique identifier for output file.
    mask_template [string; default=FALSE]:
        file path to grab a subject specific mask, a binary file that will be multipled into the final atlas. Initially used to multiply in significance threshold into a functional localizer.
    '''
    import nibabel as nib
    import numpy as np

    def get_files(subj_id, template):
        import glob
        out_list = []
        for x in glob.glob(template):
            if subj_id in x:
                out_list.append(x)
        return out_list

    for subj in subj_list:
        try:
            subj_files = get_files(subj, dir_template)
        except:
            print('could not find files for %s' % subj)
            continue
        for file in subj_files:
            if file == subj_files[0]:
                base_file = nib.load(file)
                base_data = base_file.get_data()
            else:
                base_data = base_data + nib.load(file).get_data()
        if mask_template:
            mask_file = get_files(subj, mask_template)
            if len(mask_file) != 1:
                print('No mask found for %s, SKIPPING' % subj)
                continue
            mask_data = nib.load(mask_file[0]).get_data()
            base_data = base_data * mask_data
        out_file = nib.Nifti1Image(base_data, base_file.affine, base_file.header)
        out_file.header['cal_max'] = np.max(base_data) # fix header info
        out_file.header['cal_min'] = 0 # fix header info
        try:
            nib.save(out_file, os.path.join(out_dir, subj+'_'+out_name+'.nii'))
        except:
            os.makedirs(out_dir)
            nib.save(out_file, os.path.join(out_dir, subj+'_'+out_name+'.nii'))

def extract_rois_from_atlas(subj_list, data_template, atlas_template, out_dir, num_rois=False, subj_specific_atlas=False, headers=False):
    '''
    Grab all unique values (except 0) from a 3d Nifti atlas, then extract the average of all voxels in that ROI from another timeseries.

    subj_list [list of strings]
        e.g. ['sub-03', 'sub-04']
    data_template [string]
        Can include wildcards to use with glob. Can reference all subject files, then the subject ID will be used to narrow the glob search.
    atlas_template [string]
        Can include wildcards (e.g. if each subject has their own unique atlas)
        If subj_specific_atlas is true, then subject Id will be used to narrow a glob search.
    out_dir [string]
        output directory
    num_rois [default=False; integer]
        number of unique values in the atlas. Used when subject specific atlases may or may not contain all ROIs. If not entered, the number of rois will be inferred by np.unique(). In either case, it is assumed that rois are represented by sequential integers.
    subject_specific_atlas [default=False; boolean]
        used if each subject has a unique atlas (i.e. subject-specific ROI). In this case, atlas_template will likely need a wildcard.
    headers [default=False; list of strings]
        Names of ROIs in target atlas. Will name columns in output.
        e.g. ['PC_ToMN', 'RTPJ_ToMN', 'LTPJ_ToMN']
    '''
    def get_files(subj_id, template):
        import glob
        out_list = []
        for x in glob.glob(template):
            if subj_id in x:
                out_list.append(x)
        return out_list

    import nibabel as nib
    import numpy as np
    import pandas as pd
    import os

    for subj in subj_list:
    ## Get atlas
        if subj_specific_atlas:
            atlas_path = get_files(subj, atlas_template)
            assert atlas_path, 'atlas path returned no images'
            assert len(atlas_path) == 1, 'atlas path must return one image.'
            atlas_path = atlas_path[0]
        else:
            atlas_path = atlas_template
        atlas_data = nib.load(atlas_path).get_fdata()
        atlas_name = atlas_path.split('/')[-1].split('.')[0]
        assert len(atlas_data.shape)==3, 'atlas file must be a 3d file.'
        if headers:
            assert isinstance(headers, list), 'headers must be a list'
            if num_rois:
                len(headers)==num_rois
            else:
                assert len(headers)==len(np.unique(atlas_data))-1, 'headers must have a value for every unique roi'
            out_header = pd.DataFrame(columns=headers)
        else:
            out_header = pd.DataFrame(columns=['r'+str(val) for val in list(range(1,len(np.unique(atlas_data))))])

    ## Get subject data.
        subj_files = get_files(subj, data_template)
        assert len(subj_files) > 0, 'no files found for subject'
        for file in subj_files:
            file_name = file.split('/')[-1].split('.')[0]
            print(('grabbing %s for subject %s') % (file, subj))
            subj_data = nib.load(file).get_fdata()
            assert atlas_data.shape == subj_data.shape[0:3], 'bold files and atlas file must be same dimensions'
            atlas_data = np.rint(np.ndarray.flatten(atlas_data))
            print('atlas flattened')
            subj_data = subj_data.reshape(np.prod(subj_data.shape[0:3]), subj_data.shape[-1]) # flatten to voxelxTR
            print('data reshaped')
            if num_rois:
                roi_max = num_rois
            else:
                roi_max = len(np.unique(atlas_data))-1 # -1 is because we don't want to count zero (empty data)
            temp_out = np.empty((subj_data.shape[-1], num_rois), dtype=np.float64)
            temp_out[:] = np.nan
            for idx, roi in enumerate(list(range(1,roi_max+1))):
                if atlas_data[np.where(atlas_data==roi)].size == 0: # skip if not data for ROI.
                    continue
                temp_out[:, idx] = np.mean(subj_data[np.where(atlas_data==roi)],0)
                print('averaged roi: %s' % roi)
            out_data = out_header.append(pd.DataFrame(temp_out, columns=list(out_header.columns.values)),
                                       ignore_index=True)
            try:
                out_data.to_csv(os.path.join(out_dir, subj+'_'+file_name+'_atlas-'+atlas_name+'.tsv'), sep='\t', index=False)
            except:
                os.makedirs(os.path.join(out_dir))
                out_data.to_csv(os.path.join(out_dir, subj+'_'+file_name+'_atlas-'+atlas_name+'.tsv'), sep='\t', index=False)
            print('done with file: \n %s.\n Input shape = %s\n Output shape = %s\n' % (file, subj_data.shape, out_data.shape))

def itegrate_roi2BIDs(subj_list, event_template, roi_templates, out_dir, out_name, TR):
    '''
    Extract data from roi files (generated by extract_rois_from_atlas), averaging across duraton following onset.
    Data is then combined into the BIDS event files and turned into one .csv output.

    subj_list [list of strings]
        e.g. ['sub-03', 'sub-04']
    event_template [string]
        Can include wildcards to use with glob. Can reference all event files (for all subjects), then the subject ID will be used to narrow the glob search.
    roi_templates [string, or list of strings]
        glob template to find roi files from extract_rois_from_atlas. If a list is given, then averages will be taken for columns in all ROI files, and appended as columns.
    out_dir [string]
        output directory
    out_name [string]:
        desired name for output file.
    TR [integer]:
        TR of original data, as event files should have onsets/durations in seconds (TODO - add options for this.)
    '''
    def get_files(subj_id, template):
        import glob
        out_list = []
        for x in glob.glob(template):
            if subj_id in x:
                out_list.append(x)
        return out_list

    import numpy as np
    import pandas as pd
    import os

    if not isinstance(roi_templates, (list,)): # convert any template into a list, so we can accept multiple templates.
        roi_templates = [roi_templates]
    for subj in subj_list:
        print('working on subject: %s' % subj)
        for roi_path in roi_templates:
            roi_files = sorted(get_files(subj, roi_path)) # get sorted list of roi files.
            for rfile in roi_files: # append all rois to each other
                print('grabbing roi: %s' % rfile)
                rtemp = pd.read_csv(rfile, sep='\t', index_col=None)
                runlen_TR = len(rtemp.index) # get number of rows to get TR per run.
                if rfile == roi_files[0]: # append all run data as new rows.
                    rdata = rtemp.copy()
                else:
                    rdata = rdata.append(rtemp, ignore_index=True)
            if roi_path == roi_templates[0]: # append multiple roi entries together, as new columns.
                all_rdata = rdata.copy()
            else:
                all_rdata = pd.concat([all_rdata, rdata], axis=1, sort=False)

        event_files = sorted(get_files(subj, event_template))
        assert len(event_files) == len(roi_files), 'mismatch in # of event files and roi files.'
        for idx, efile in enumerate(event_files): # append all event files to each other.
            print('grabbing event file: %s' % efile)
            etemp = pd.read_csv(efile, sep='\t', index_col=None)
            etemp['onset'] = etemp['onset']/TR + runlen_TR*idx # add # of TRs/run to onset, to get absolute onset.
            etemp['duration'] = etemp['duration']/TR
            if efile == event_files[0]:
                edata = etemp.copy()
            else:
                edata = edata.append(etemp, ignore_index=True)
        # get location of all roi data, so we can compute an average across them.
        roi_locs = [list(range(int(onset), int(onset)+int(edata['duration'][idx]))) for idx, onset in enumerate(edata['onset'])]
        r_avg = pd.DataFrame(np.empty((len(roi_locs), len(all_rdata.columns.values)), dtype=np.float),
                             columns = all_rdata.columns.values)
        r_avg[:] = np.nan
        # compute average for each roi
        for roi in all_rdata.columns.values:
            print('averaging roi: %s' % roi)
            r_avg[roi] = [np.mean(all_rdata[roi][r_locs]) for r_locs in roi_locs]
        edata = pd.concat([edata, r_avg], axis=1, sort=False)
        # combine subject data
        if subj == subj_list[0]:
            out_data = edata.copy()
        else:
            out_data = pd.concat([out_data, edata], axis=0, sort=False)
    # save data
    try:
        out_data.to_csv(os.path.join(out_dir, out_name+'.tsv'), sep='\t', index=False)
    except:
        os.makedirs(os.path.join(out_dir))
        out_data.to_csv(os.path.join(out_dir, out_name+'.tsv'), sep='\t', index=False)
