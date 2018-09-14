def extract_PAG(p_thresh_list):
    '''
    The workflow takes the following as input to wf.inputs.inputspec
    Input [Mandatory]:
        ~~~~~~~~~~ Set as part of function call:
        p_thresh_list: list of floats representing p thresholds. Applied to resdiduals. ITERABLE.
            e.g. [95, 97.5, 99.9]

    ~~~~~~~~~~~ Set through inputs.inputspec
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

    TODO
    '''
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function

    ################## Setup workflow.
    # grandmean_wf = pe.Workflow(name='make_ref_img')
    # inputspec = pe.Node(IdentityInterface(
    #     fields=['subject_list', 'template']),
    #              name='inputspec')
    # outputspec = pe.Node(IdentityInterface(
    #     fields=['mean_img']),
    #                     name='outputspec')
    #
    # def get_files(subject_list, template):
    #     import glob
    #     out_list = []
    #     for x in glob.glob(list(template.values())[0]):
    #         if any(subj in x for subj in subject_list):
    #             out_list.append(x)
    #     return out_list
    #
    # get_imgs = pe.Node(Function(
    #     input_names=['subject_list', 'template'],
    #     output_names=['out_list'],
    #     function=get_files),
    #                    name='get_imgs')
    #
    #    infosource = pe.Node(IdentityInterface(fields=['p_thresh']),
    #                name='infosource')
    #    infosource.iterables = [('p_thresh', p_thresh_list)]

       def mask_thresh_clust_list(file_list, mask_file, thresh=95, cluster_k=50):
           import nibabel as nib
           import numpy as np
           from scipy.ndimage import label, zoom
           mask = nib.load(mask_file)
           temp = nib.load(file_list[0]) # load reference image.
           interp_dims = np.array(temp.shape[0:3])/np.array(mask.shape)
           mask = zoom(mask.get_data(), interp_dims.tolist()) # interpolate mask to native space.
           imgs = np.empty((temp.shape[0], #create empty array
                            temp.shape[1],
                            temp.shape[2], len(file_list)))
           for idx, file in enumerate(file_list): # iterate through residuals.
               img = nib.load(file).get_data() # grab data
               img[img[...,0]==0] = np.nan # threshold voxels outside brain, nan used to avoid problems with percentiles.
               img = np.nanmean(img, axis=3) # average over time.
               img[mask!=1] = np.nan # mask
               img[img < np.nanpercentile(img, thresh)] = np.nan #threshold residuals.
               label_map, n_labels = label(np.nan_to_num(img)) # label remaining voxels.
               for label_ in range(1, n_labels+1): # addition to match labels, which are base-1.
                   lab_val = 1
                   if np.sum(label_map==label_) > cluster_k:
                       imgs[...,idx][label_map==label_] = lab_val # zero any clusters below cluster threshold.
                       lab_val = lab_val+1

           return imgs # imgs is 4d, containing all subjects, has labels numbered in each 3d array.

       def mask_thresh_clust(file, mask_file, thresh=95, cluster_k=50):
           import nibabel as nib
           import numpy as np
           from jt_7t import fit_mask
           from scipy.ndimage import label, zoom
           img = np.nanmean(nib.load(file).get_data(), axis=3) # grab data, average along time.
           mask = fit_mask(mask_file, file)
           img[mask!=1] = np.nan # mask
           img[img < np.nanpercentile(img, thresh)] = np.nan #threshold residuals.
           out_labeled = np.empty((img.shape[0], img.shape[1],img.shape[2]))
           label_map, n_labels = label(np.nan_to_num(img)) # label remaining voxels.
           lab_val = 1
           for label_ in range(1, n_labels+1): # addition to match labels, which are base-1.
               if np.sum(label_map==label_) >= cluster_k:
                   print('found a cluster!')
                   out_labeled[label_map==label_] = lab_val # zero any clusters below cluster threshold.
                   lab_val = lab_val+1
           return out_labeled


           # grab files.




# Grab all files.
# For each file:
    # get data (nibabel) DONE
    #average along 4d. DONE
    # mask with search region. DONE
    # grab data below 95 percentile of residuals. DONE
    # get clusters of 50. DONE
    # define regions as clusters (maybe) DONE

# get the average from this first pass, this is our template.
    # save this somewere, as aqueduct_template.nii

# iterate through each file again.
    # get data, mask, threshold (at 95, 97.5, 99.9), get clusters.
    # for each cluster:
        # correlate with aqueduct_template.nii
    # find max correlation.
    # save corresponding cluster.



# To create template for PAG, use volume-based method (FNIRT in fsl)
