def create_aqueduct_template(subj_list, p_thresh_list, template, work_dir, out_dir, space_mask):
    '''
    The workflow takes the following as input to wf.inputs.inputspec
    Input [Mandatory]:
        subj_list: list of subject IDs
            e.g. [sub-001, sub-002]

        p_thresh_list: list of floats representing p thresholds. Applied to resdiduals.
            e.g. [95, 97.5, 99.9]

        template: string to identify all PAG aqueduct files (using glob).
            e.g. template = '/home/neuro/func/sub-001/sigmasquareds.nii.gz'
                The template can identify a larger set f files, and the subject_list will grab a subset.
                    e.g. The template may grab sub-001, sub-002, sub-003 ...
                    But if the subject_list only includes sub-001, then only sub-001 will be used.
                    This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

        work_dir: string, denoting path to working directory.

        out_dir: string, denoting output directory (results saved to work directory and output)

        space_mask: string, denoting path to PAG search region mask.
    Output:
        /subj_cluster/sub-xxx_sigmasquare_clusts.nii.gz
            folder within work_dir, listing nii.gz images with all clusters t the specified p thresholds and cluster extents (default = 50 only)
        /templates/sub-xxx_aqueduct_template.nii.gz
            all subject aqueduct templates output as nii.gz files
        /templates/MEAN_aqueduct_template.nii.gz
            average of all aqueduct templates, which was used to help converge on the correct cluster.
        /templates/report.csv
            report on output, listing subject, threshold used, cluster, corelation with the average, and how many iterations it took to settle on an answer. Results where corr < .3 are flagged.
    '''
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import os
    from jtnipyutil.util import files_from_template, clust_thresh, mask_img

    for subj in subj_list: # For each subject, create aqueduct template file wtih all thresholded clusters.
        print('creating aqueduct template for %s' % subj)
        try:
            img_info = nib.load(files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
        except:
            img_file  = files_from_template(subj, template)[0]
            img_info = nib.load(img_file)
            img = mask_img(img_file, space_mask, out_format = 'array') # loading done here. Slow.
            # img = np.nanmean(img, axis=3) # Average data along time.
            for thresh in p_thresh_list:
                img_labeled = clust_thresh(img, cluster_k=[50], thresh = thresh)
                if thresh == p_thresh_list[0]:
                    all_labeled = img_labeled[..., np.newaxis]
                else:
                    all_labeled = np.append(all_labeled, img_labeled[..., np.newaxis], axis=3) # stack thresholds along 4th dim.
            pag_img = nib.Nifti1Image(all_labeled, img_info.affine, img_info.header)
            pag_img.header['cal_max'] = np.max(all_labeled) # fix header info
            pag_img.header['cal_min'] = 0 # fix header info
            try:
                nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
                nib.save(pag_img, os.path.join(out_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
            except:
                os.makedirs(os.path.join(work_dir, 'subj_clusts'))
                os.makedirs(os.path.join(out_dir, 'subj_clusts'))
                nib.save(pag_img, os.path.join(work_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))
                nib.save(pag_img, os.path.join(out_dir, 'subj_clusts', subj+'_sigmasquare_clusts.nii.gz'))

    ## gather all subjects clusters/thresholds into a 5d array. ##########################################
    for subj in subj_list:
        img_file = files_from_template(subj, os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))
        print(('getting data from %s') % img_file[0])
        img = nib.load(img_file[0]).get_data()
        if subj == subj_list[0]:
            all_subj_data = img[..., np.newaxis]
        else:
            all_subj_data = np.append(all_subj_data, img[...,np.newaxis], axis=4)

    ## get mean across defaults: threshold (95) and cluster (1) ##########################################
    # This establishes a template to judge which threshold fits it best.
    # Average is across all subjects.
    aq_template = np.copy(all_subj_data[...,0,:])
    aq_template[aq_template != 1] = 0
    aq_template = np.mean(aq_template, axis=3)
    # set up report.
    aq_report = pd.DataFrame(columns=['sub', 'thresh', 'clust', 'corr', 'iter', 'FLAG'], data={'sub':subj_list, 'iter':[0]*len(subj_list), 'FLAG':['']*len(subj_list)})
    aq_report = aq_report.set_index('sub')
    while True:
        aq_report['iter'] = aq_report['iter'] + 1
        new_template = np.zeros(list(all_subj_data.shape[0:3]) + [all_subj_data.shape[-1]])
        for subj_idx, subj in enumerate(subj_list):
            corr_val = -101
            for thresh_idx, thresh in enumerate(p_thresh_list):
                if np.max(all_subj_data[..., thresh_idx, subj_idx]) > 0:
                    for cluster in np.unique(all_subj_data[...,thresh_idx, subj_idx]):
                        if cluster == 0:
                            continue
                        else:
                            print(('checking sub %s, thresh %s, clust %s') % (subj, thresh, cluster))
                            test_array = np.copy(all_subj_data[...,thresh_idx, subj_idx]) # binarize array being tested.
                            test_array[test_array != cluster] = 0
                            test_array[test_array == cluster] = 1
                            clust_corr = np.corrcoef(np.ndarray.flatten(aq_template), # correlate with group mean.
                                                      np.ndarray.flatten(test_array))[0,1]
                            if clust_corr > corr_val:
                                print(('sub %s, thresh %s, clust %s, corr =  %s (prev max corr = %s)') %
                                      (subj, thresh, cluster, clust_corr, corr_val))
                                aq_report.at[subj, 'thresh'] = thresh
                                aq_report.at[subj, 'clust'] = cluster
                                aq_report.at[subj, 'corr'] = clust_corr
                                if clust_corr < .3:
                                    aq_report.at[subj, 'FLAG'] = 'CHECK'
                                else:
                                    aq_report.at[subj, 'FLAG'] = ''
                                new_template[...,subj_idx] = test_array
                                corr_val = clust_corr

        if np.array_equal(np.around(aq_template, 4), np.around(np.mean(new_template, axis=3), 4)):
            print('We have converged on a stable average for aq_template.')
            break
        else:
            aq_template = np.mean(new_template, axis=3)
            print('new aq_template differs from previous iteration. Performing another iteration.')

    for img_idx in range(0, new_template.shape[-1]):
        print(('Saving aqueduct for %s') % (subj_list[img_idx]))
        img_info = nib.load(files_from_template(subj_list[img_idx], os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
        subj_temp = nib.Nifti1Image(new_template[...,img_idx], img_info.affine, img_info.header)
        subj_temp.header['cal_max'] = 1 # fix header info
        subj_temp.header['cal_min'] = 0 # fix header info
        try:
            nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
            nib.save(subj_temp, os.path.join(out_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
        except:
            os.makedirs(os.path.join(work_dir, 'templates'))
            os.makedirs(os.path.join(out_dir, 'templates'))
            nib.save(subj_temp, os.path.join(work_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))
            nib.save(subj_temp, os.path.join(out_dir, 'templates', subj_list[img_idx]+'_aqueduct_template.nii.gz'))

    print('Saving aqueduct mean template.')
    img_info = nib.load(files_from_template(subj_list[0], os.path.join(work_dir, 'subj_clusts', '*_sigmasquare_clusts.nii.gz'))[0])
    aq_temp_img = nib.Nifti1Image(aq_template, img_info.affine, img_info.header)
    aq_temp_img.header['cal_max'] = 1 # fix header info
    aq_temp_img.header['cal_min'] = 0 # fix header info
    nib.save(aq_temp_img, os.path.join(work_dir, 'templates', 'MEAN_aqueduct_template.nii.gz'))
    nib.save(aq_temp_img, os.path.join(out_dir, 'templates', 'MEAN_aqueduct_template.nii.gz'))

    print('Saving report')
    aq_report.to_csv(os.path.join(work_dir, 'templates', 'report.csv'))
    aq_report.to_csv(os.path.join(out_dir, 'templates', 'report.csv'))

def make_PAG_masks(subj_list, data_template, gm_template, work_dir, out_dir, gm_thresh = .5, gm_spline=3, dilation_r=2, x_minmax=False, y_minmax=False, z_minmax=False):
    '''
    subj_list: list of subject IDs
        e.g. [sub-001, sub-002]

    data_template: string to identify all PAG aqueduct files (using glob).
        e.g. data_template = os.path.join(work_dir, 'templates', '*_aqueduct_template.nii.gz')
            The template can identify a larger set of files, and the subject_list will grab a subset.
                e.g. The template may grab sub-001, sub-002, sub-003 ...
                But if the subject_list only includes sub-001, then only sub-001 will be used.
                This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

    gm_template: string to identify all PAG aqueduct files (using glob).
        e.g. gm_template = os.path.join('work_dir, 'gm', '*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')

    work_dir: string, denoting path to working directory.

    out_dir: string, denoting output directory (results saved to work directory and output)

    gm_thresh: float specifying the probability to threshold gray matter mask.
        Default: .5

    gm_spline: integer specifying the spline order to use to reslice gray matter to native space (if necessary)
        Default: 3

    dilation_r: integer specifying the number of voxels to dilate the aqueduct.
        Default: 2

    x_minmax: list of 2 integers, denoting min and max X voxels to include in mask.
        e.g. [0,176]. Defaults to full range of PAG aqueduct image.

    y_minmax: list of 2 integers, denoting min and max Y voxels to include in mask.
            e.g. [82,100]. Defaults to full range of PAG aqueduct image.

    z_minmax: list of 2 integers, denoting min and max Z voxels to include in mask.
        e.g. [58,176]. Defaults to full range of PAG aqueduct image.

    '''
    import nibabel as nib
    import numpy as np
    from skimage.transform import resize
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import ball
    import os
    from jtnipyutil.util import files_from_template, clust_thresh, mask_img

    for subj in subj_list:
        print('making PAG mask for subject: %s' % subj)
        # get aqueduct.
        img_file = nib.load(files_from_template(subj, data_template)[0])
        img = img_file.get_data()
        # get gray matter, binarize at threshold.
        gm_file = nib.load(files_from_template(subj, gm_template)[0])
        gm_img = gm_file.get_data()
        if img.shape[0:3] !=  gm_img.shape[0:3]:
            gm_img = resize(gm_img, img.shape[0:3], order=gm_spline, preserve_range=True)
        gm_img[np.where(gm_img < gm_thresh)] = 0
        # create mask for PAG location.
        if not x_minmax:
            x_minmax = [0,list(img.shape)[0]]
        if not y_minmax:
            y_minmax = [0,list(img.shape)[1]]
        if not z_minmax:
            z_minmax = [0,list(img.shape)[2]]
        loc_mask = np.zeros(list(img.shape))
        loc_mask[x_minmax[0]:x_minmax[1],
             y_minmax[0]:y_minmax[1],
             z_minmax[0]:z_minmax[1]] = 1
        # dilate and subtract original aqueduct.
        pag = binary_dilation(img, ball(dilation_r)).astype(img.dtype) - img
        pag = pag*gm_img # multiply by thresholded gm probability mask.
        pag = pag*loc_mask # threshold by general PAG location cutoffs.
        pag_file = nib.Nifti1Image(pag, img_file.affine, img_file.header)
        try:
            nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii'))
            nib.save(pag_file, os.path.join(out_dir, 'pag_mask', subj+'_pag_mask.nii'))
        except:
            os.makedirs(os.path.join(work_dir, 'pag_mask'))
            os.makedirs(os.path.join(out_dir, 'pag_mask'))
            nib.save(pag_file, os.path.join(work_dir, 'pag_mask', subj+'_pag_mask.nii'))
            nib.save(pag_file, os.path.join(out_dir, 'pag_mask', subj+'_pag_mask.nii'))

def create_DARTEL_wf(subj_list, file_template, work_dir, out_dir):
    '''
    Aligns all images to a template (average of all images), then warps images into MNI space (using an SPM tissue probability map, see https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf, section 25.4).

    subj_list: list of subject IDs
        e.g. [sub-001, sub-002]

    file_template: string to identify all files to align (using glob).
        e.g. file_template = os.path.join(work_dir, 'pag_mask', '*_pag_mask.nii')
            The template can identify a larger set of files, and the subject_list will grab a subset.
                e.g. The template may grab sub-001, sub-002, sub-003 ...
                But if the subject_list only includes sub-001, then only sub-001 will be used.
                This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)

    work_dir: string, denoting path to working directory.

    out_dir: string, denoting output directory (results saved to work directory and output)
    '''
    import nibabel as nib
    import numpy as np
    from nipype.interfaces.spm.preprocess import DARTEL, CreateWarped
    from nipype.interfaces.io import DataSink
    import nipype.pipeline.engine as pe
    import os
    from jtnipyutil.util import files_from_template
    # set up workflow.
    DARTEL_wf = pe.Workflow(name='DARTEL_wf')
    DARTEL_wf.base_dir = work_dir

    # get images
    images = files_from_template(subj_list, file_template)

    # set up DARTEL.
    dartel = pe.Node(interface=DARTEL(), name='dartel')
    dartel.inputs.image_files = [images]

    dartel_warp = pe.Node(interface=CreateWarped(), name='dartel_warp')
    dartel_warp.inputs.image_files = images
    #     warp_data.inputs.flowfield_files = # from inputspec

    ################## Setup datasink.
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')
    sinker.inputs.base_directory = out_dir

    DARTEL_wf.connect([(dartel, dartel_warp, [('dartel_flow_fields', 'flowfield_files')]),
                       (dartel, sinker, [('final_template_file', 'avg_template'),
                                        ('template_files', 'avg_template.@template_stages'),
                                        ('dartel_flow_fields', 'dartel_flow')]),
                       (dartel_warp, sinker, [('warped_files', 'warped_PAG')])])

    return DARTEL_wf

def setup_DARTEL_warp_wf(subj_list, data_template, warp_template, work_dir, out_dir):
    '''
    subj_list: list of strings for each subject
        e.g. ['sub-001', 'sub-002', 'sub-003']
    data_template: string to identify all data files (using glob).
            e.g. template = '/home/neuro/data/rest1_AROMA/nosmooth/sub-*/model/sub-*/_modelestimate0/res4d.nii.gz'
                The template can identify a larger set of files, and the subject_list will grab a subset.
                    e.g. The template may grab sub-001, sub-002, sub-003 ...
                    But if the subject_list only includes sub-001, then only sub-001 will be used.
                    This means the template can overgeneralize, but specific subjects can be easily excluded (e.g. for movement)
    warp_template: string to identify all dartel flowfield files (using glob).
        same as above.
        Dartel flowfield files are made by create_DARTEL_wf,
            also see jtnipyutil.fsmap.make_PAG_masks, and jtnipyutil.fsmap.create_aqueduct_template
    work_dir: string naming directory to store work.
    out_dir: string naming directory for output.
    '''
    import os
    import nibabel as nib
    import numpy as np
    import nipype.pipeline.engine as pe
    from nipype import IdentityInterface
    from nipype.interfaces.io import DataSink
    from nipype.interfaces.utility.wrappers import Function
    from nipype.interfaces.spm.preprocess import CreateWarped
    from jtnipyutil.util import files_from_template

    # create working directory if necessary.
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set up data warp workflow
    apply_warp_wf = pe.Workflow(name='apply_warp_wf')
    apply_warp_wf.base_dir = work_dir

    # set up file lists
    inputspec = pe.Node(IdentityInterface(
        fields=['file_list',
                'warp_list']),
                       name='inputspec')
    inputspec.inputs.file_list = files_from_template(subj_list, data_template)
    inputspec.inputs.warp_list = files_from_template(subj_list, warp_template)

    # rename files, as names are often indistinguishable (e.g. res4d.nii.gz)
    def rename_list(in_list):
        import nibabel as nib
        import os
        out_list = []
        for file in in_list:
            file_in = nib.load(file)
            nib.save(file_in, os.path.join(os.getcwd(), '_'.join(file.split('/')[-3:])))
            out_list.append(os.path.join(os.getcwd(), '_'.join(file.split('/')[-3:])))
        return out_list

    rename = pe.Node(Function(
        input_names=['in_list'],
        output_names=['out_list'],
            function=rename_list),
                    name='rename')

    # dartel warping node.
    warp_data = pe.Node(interface=CreateWarped(), name='warp_data')
#     warp_data.inputs.image_files = # from inputspec OR gunzip
#     warp_data.inputs.flowfield_files = # from inputspec

    sinker = pe.Node(DataSink(), name='sinker')
    sinker.inputs.base_directory = out_dir

    # check if unzipping is necessary.
    apply_warp_wf.connect([(inputspec, rename, [('file_list', 'in_list')]),
                           (inputspec, warp_data, [('warp_list', 'flowfield_files')]),
                           (warp_data, sinker, [('warped_files', 'warped_files')])])
    if any('nii.gz' in file for file in files_from_template(subj_list, data_template)):
        from nipype.algorithms.misc import Gunzip
        gunzip = pe.MapNode(interface=Gunzip(), name='gunzip', iterfield=['in_file'])
        apply_warp_wf.connect([(rename, gunzip, [('out_list', 'in_file')]),
                               (gunzip, warp_data, [('out_file', 'image_files')])])
    else:
        apply_warp_wf.connect([(rename, warp_data, [('out_list', 'image_files')])])
    return apply_warp_wf

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


def segment_and_unroll_PAG(PAG_file, con_file, con_name, out_dir, thresh = .2):
    '''
    Grab PAG mask, 2nd level contrast, then unwrap the PAG along its length, producing a map of
    left/right dorsal and ventral strips, and the dorsomedial strip.

    PAG_file [string, path and file name]: Dartel template (e.g. Template_6) from jtnipyutil.fsmap.create_DARTEL_wf.
        e.g. '/home/neuro/workdir/2019-12-04_PAG_lvl2/template/Template_6.nii')
    con_file [string, path and file name]: beta-weight .nii.gz file from a level 2 contrast.
        e.g. '/home/neuro/workdir/2019-12-04_PAG_lvl2/data/dartel/nosmooth/_contrast_2_speech_prep/randomise_raw_tstat1.nii.gz'
    con_name [string]: name of contrast.
        e.g. 'speech_prep'
    out_dir [string, path]: output direction.
        e.g. '/home/neuro/workdir/2019-12-04_PAG_lvl2/output'
    thresh [float, default = .2]: probability threshold for PAG_file mask.
    '''
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    import nibabel as nib
    import numpy as np
    import os, glob
    import matplotlib.pyplot as plt

    def cart2pol(x, y, z):
        import numpy as np
        theta = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        z = z
        return (theta, rho, z)

    # make binary mask of PAG
    roi_data = nib.load(PAG_file).get_data()
    roi_xyz = np.where(roi_data >= thresh) # grab coordinates within mask.
    roi_data[np.where(roi_data < thresh)] = 0 # zero everything below the threshold
    roi_xyz_2d = np.array([[x, y, z] for x, y, z in zip(roi_xyz[0], roi_xyz[1], roi_xyz[2])]) # transform 3d coordinates into a 2d array

    # PCA into polar coordinates within PAG.
    pca = PCA().fit(roi_xyz_2d) # First dim will be along the length of the PAG, as it carries the most variance.
    pca_score = pca.transform(roi_xyz_2d)

    # Transform to polar coordinates
    theta, rho, z = cart2pol(pca_score[:,1]*-1, pca_score[:,2], pca_score[:,0]) # NOTE: # xdim flipped, to give dorsomedial at center, unzipping at ventromedial.
    pag_degree = theta*180/np.pi

    # Cluster intro strips.
    agglomo_clust_degree_pag = AgglomerativeClustering(n_clusters = 5,
                                                affinity='l1',
                                                linkage='average').fit(np.array([pag_degree, z]).transpose())
    ## plot segmentation.
    # First, order the segments
    cluster_means = []
    cluster_ids = []
    for label in np.unique(agglomo_clust_degree_pag.labels_):
        cluster_means.append(np.mean(pag_degree[agglomo_clust_degree_pag.labels_==label]))
        cluster_ids.append(label)
    cluster_order = sorted(zip(cluster_means, cluster_ids))

    # then label.
    labdict = {'L-ventral':cluster_order[0][1],
               'L-dorsal':cluster_order[1][1],
               'dorsomedial':cluster_order[2][1],
               'R-dorsal':cluster_order[3][1],
               'R-ventral':cluster_order[4][1],
               }
    cdict = {'dorsomedial': 'teal',
            'L-dorsal': 'royalblue',
            'R-dorsal': 'salmon',
            'L-ventral': 'darkred',
            'R-ventral': 'blueviolet'}

    # Create figure of slice through PAG
    fig, ax = plt.subplots()
    for lab in labdict:
        ix = np.where(agglomo_clust_degree_pag.labels_ == labdict[lab])
        ax.scatter(pca_score[ix,1], pca_score[ix,2], c=cdict[lab], label=lab, s=200, alpha=.6)
    plt.legend(loc='upper right', fontsize ='small', bbox_to_anchor=(1.26,1.05))
    plt.savefig(os.path.join(out_dir, 'PAG_slice.png'))
    plt.clf()

    # Create figure of PAG unrolled.
    fig, ax = plt.subplots()
    for lab in labdict:
        ix = np.where(agglomo_clust_degree_pag.labels_ == labdict[lab])
        ax.scatter(pag_degree[ix], z[ix], c=cdict[lab], label=lab, s=200, alpha=.6) # pag_degree flipped to put left PAG on left side of graph.
    plt.legend(loc='upper right', fontsize ='small', bbox_to_anchor=(1.26,1.05))
    plt.savefig(os.path.join(out_dir, 'PAG_unrolled.png'))
    plt.clf()

    # Get beta weights within PAG mask.
    PAG_wholebrain = nib.load(con_file).get_data()
    PAG_masked = PAG_wholebrain[np.where(roi_data >= thresh)]

    # Create figure of contrast beta estimates within PAG mask.
    plt.scatter(pag_degree*-1, z,c=PAG_masked, cmap=plt.cm.coolwarm, s=200, alpha=.6)
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, con_name+'in_PAG_unrolled.png'))
    plt.clf()

    # Fill back in the kmeans labels into the original mask.
    for lab in np.unique(agglomo_clust_degree_pag.labels_):
        label_xyz = roi_xyz_2d[agglomo_clust_degree_pag.labels_ == lab]
        x_coord = [label_xyz[:][coord][0] for coord in range(len(label_xyz))]
        y_coord = [label_xyz[:][coord][1] for coord in range(len(label_xyz))]
        z_coord = [label_xyz[:][coord][2] for coord in range(len(label_xyz))]
        roi_data[x_coord, y_coord, z_coord] = lab+1
    # save and output new PAG masks.
    roi_data_out = nib.Nifti1Image(roi_data, nib.load(PAG_file).affine, nib.load(PAG_file).header)
    prefix = PAG_file.split('/')[-1].split('.nii')[0]+'_agglomCluster'
    for k_out in range(5): # save separate kmean clusters.
        roi_data_k = np.copy(roi_data)
        roi_data_k[roi_data != k_out+1] = 0
        roi_data_k[roi_data == k_out+1] = 1
        roi_kdata_out = nib.Nifti1Image(roi_data_k, nib.load(PAG_file).affine, nib.load(PAG_file).header)
        nib.save(roi_kdata_out, os.path.join(out_dir, prefix + '-' + str(k_out+1)+'.nii'))
    # save full ROI.
    nib.save(roi_data_out, os.path.join(out_dir, prefix + ".nii"))
