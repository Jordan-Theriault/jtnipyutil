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

def fit_mask(mask_file, ref_file, work_dir = '', out_format = 'file' ):
    '''
    Fits a mask file to the space of a reference image.
    Input [Mandatory]:
        mask_file: path to a nifti mask file to be refit to reference space.
        ref_file: path to a nifti file in the reference space. Can be 3d or 4d.
            3d space is the reference dimension.
        work_dir: [default = ''] path to directory to save masked file. Required if out_format = 'file'.
        out_format: [default = 'file'] Options are 'file', or 'array'.
    Output
        out_mask: Either a nifti file, or a np array, depending on out_format.
    '''
    import numpy as np
    import nibabel as nib
    import os.path
    from scipy.ndimage import zoom
    mask = nib.load(mask_file)
    mask_name = '_'+mask_file.split('/')[-1].split('.')[0]
    ref = nib.load(ref_file)
    if mask.shape != ref.shape[0:3]:
        interp_dims = np.array(ref.shape[0:3])/np.array(mask.shape)
        data = zoom(mask.get_data(), interp_dims.tolist()) # interpolate mask to native space.
    else:
        print('mask is already in reference space!')

    if out_format == 'file':
        assert (work_dir != ''), 'You must give a value for work_dir'
        out_mask = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(out_mask, os.path.join(work_dir, mask_name + '_fit.nii.gz'))
        out_mask = os.path.join(work_dir, mask_name + '_fit.nii.gz')
    else:
        assert (out_format == 'array'), 'out_format is neither file, or array.'
        out_mask = data

    return out_mask

def mask_img(img_file, mask_file, work_dir = '', out_format = 'file'):
    '''
    Fits a mask file to the space of a reference image.
    Input [Mandatory]:
        img_file: path to a nifti file to be masked. Can be 3d or 4d.
        mask_file: path to a nifti mask file. Does not need to match dimensions of img_file
        work_dir: [default = ''] path to directory to save masked file. Required if out_format = 'file'.
        out_format: [default = 'file'] Options are 'file', or 'array'.
    Output
        out_img: Either a nifti file, or a np array, depending on out_format.
    '''
    import numpy as np
    import nibabel as nib
    import os.path
    from scipy.ndimage import zoom
    print(('loading mask file: %s') % mask_file)
    mask = nib.load(mask_file)
    mask_name = '_'+mask_file.split('/')[-1].split('.')[0]
    img_name = img_file.split('/')[-1].split('.')[0]
    print(('loading img file: %s') % img_file)
    img = nib.load(img_file) # grab data
    data = nib.load(img_file).get_data() # grab data
    if mask.shape != data.shape[0:3]:
        interp_dims = np.array(data.shape[0:3])/np.array(mask.shape)
        mask = zoom(mask.get_data(), interp_dims.tolist()) # interpolate mask to native space.
    else:
        mask = mask.get_data()

    data[mask!=1] = np.nan # mask

    if out_format == 'file':
        assert (work_dir != ''), 'You must give a value for work_dir'
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
        print(('looking for clusters > %s voxels') % cluster_k)
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
