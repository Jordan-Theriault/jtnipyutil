
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
        data[mask!=1] = np.nan # mask
    else:
        assert(inclu_exclu == 'exclusive'), 'mask must be either inclusive or exclusive'
        data[mask==1] = np.nan # mask

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

def combine_runs(runsecs, subj, out_folder, runs=False, bold_template = False, bmask_template = False, task_template = False, conf_template = False):
    '''
    Combines bold, task, or confound files in fmriprep folder, ASUMMNGS BIDS FOLDER STRUCTURE.
    Also can create an inclusive mask, keeping only voxels shared across all runs.
    For task, will also create a run_onset column, which lists onsets (in seconds) relative to the start of the run. The normal 'onset' column gives a unique onset value for each event.
    For confounds, ICAAroma components will have the run number appended to the right. So AROMAAggrComp28 becomes AROMAAggrComp28_r1.
    Input [Mandatory]:
        runsecs: integer, listing number of seconds per run.
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
        return out_list

    # append all bold data.
    if bold_template:
        if runs:
            file_list = list(get_filelist(subj, bold_template)[i] for i in runs)
        else:
            file_list = get_filelist(subj, bold_template)
        for file in file_list:
            if file == file_list[0]:
                ref = nib.load(file) # get header info if first file.
                out_data = nib.load(file).get_data()
            else:
                run_data = nib.load(file).get_data()
                out_data = np.append(out_data, run_data, axis=3)
        out_boldname = file.partition('/func/')[-1].partition('run-')[0] + file.partition('run-')[-1][3:]
        out_file = nib.Nifti1Image(out_data, ref.affine, ref.header)
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
                out_bmaskname = bmask.partition('/func/')[-1].partition('run-')[0] + bmask.partition('run-')[-1][3:]
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
            task_list = get_filelist(subj, task_template)
        for idx, tfile in enumerate(task_list):
            if tfile == task_list[0]:
                out_taskname = tfile.partition('/func/')[-1].partition('run-')[0] + tfile.partition('run-')[-1][3:]
                out_tdata = pd.read_csv(tfile, sep='\t', index_col=None) # start the dataframe if first file.
                out_tdata['run_onset'] = out_tdata['onset']
            else:
                run_tdata = pd.read_csv(tfile, sep='\t', index_col=None)
                run_tdata['run_onset'] = run_tdata['onset']
                run_tdata['onset'] = run_tdata['onset'] + idx*runsecs
                out_tdata = out_tdata.append(run_tdata, ignore_index = True)
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
                    if 'AROMAAggr' in col:
                        out_cdata = out_cdata.rename(columns={col: col+'_r'+str(idx+1)})
                    if 'NonSteadyStateOutlier' in col:
                        out_cdata = out_cdata.rename(columns={col: col+'_r'+str(idx+1)})
            else:
                run_cdata = pd.read_csv(cfile, sep='\t', index_col=None)
                for col in run_cdata.columns:
                    if 'AROMAAggr' in col:
                        run_cdata = run_cdata.rename(columns={col: col+'_r'+str(idx+1)})
                    if 'NonSteadyStateOutlier' in col:
                        out_cdata = out_cdata.rename(columns={col: col+'_r'+str(idx+1)})
                out_cdata = out_cdata.append(run_cdata, ignore_index = True, sort=False)
        out_cdata[out_cdata.isna()] = 0
        out_confname = cfile.partition('/func/')[-1].partition('run-')[0] + cfile.partition('run-')[-1][3:]
        out_cdata.to_csv(os.path.join(out_folder, out_confname), sep='\t', index=False)
