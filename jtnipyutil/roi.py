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

def hemi_split(roi_path, out_dir):
    '''
    Splits a mask at the X midpoint, and outputs both hemispheres as indendepent masks.

    roi_path [string, with wildcard] Path to all rois to split.
    out_dir [string] Directory to save output to.
    '''
    import glob, os
    import nibabel as nib
    import numpy as np


    for roi in roi_path:
        print('grabbing:', roi)
        roi_dat = nib.load(roi).get_fdata()
        roi_dat[roi_dat>0] = 1 # binarize, in case of probabalistic masks.
        roi_loc = np.where(roi_dat==1)

        for hem in ['L_', 'R_']:
            print('generating', hem, 'hemisphere')
            roi_out = np.zeros(roi_dat.shape)
            if hem == 'L_':
                hem_roi = [roi_loc[0][roi_loc[0]<roi_dat.shape[0]/2],
                           roi_loc[1][roi_loc[0]<roi_dat.shape[0]/2],
                           roi_loc[2][roi_loc[0]<roi_dat.shape[0]/2]]
            if hem == 'R_':
                hem_roi = [roi_loc[0][roi_loc[0]>roi_dat.shape[0]/2],
                           roi_loc[1][roi_loc[0]>roi_dat.shape[0]/2],
                           roi_loc[2][roi_loc[0]>roi_dat.shape[0]/2]]

            roi_out[tuple(hem_roi)] = nib.load(roi).get_fdata()[tuple(hem_roi)] # return original mask values, to preserb probabalistic masks.
            nib.save(nib.Nifti1Image(roi_out, nib.load(roi).affine, nib.load(roi).header),
                     os.path.join(out_dir, hem+roi.split('/')[-1]))

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
            if file == subj_files[0]:
                atlas_flat = np.rint(np.ndarray.flatten(atlas_data))
                print('atlas flattened')
            subj_data = subj_data.reshape(np.prod(subj_data.shape[0:3]), subj_data.shape[-1]) # flatten to voxelxTR
            print('data reshaped')
            if num_rois:
                roi_max = num_rois
            else:
                roi_max = len(np.unique(atlas_flat))-1 # -1 is because we don't want to count zero (empty data)
            temp_out = np.empty((subj_data.shape[-1], num_rois), dtype=np.float64)
            temp_out[:] = np.nan
            for idx, roi in enumerate(list(range(1,roi_max+1))):
                if atlas_flat[np.where(atlas_flat==roi)].size == 0: # skip if not data for ROI.
                    continue
                temp_out[:, idx] = np.mean(subj_data[np.where(atlas_flat==roi)],0)
                print('averaged roi: %s' % roi)
            out_data = out_header.append(pd.DataFrame(temp_out, columns=list(out_header.columns.values)),
                                       ignore_index=True)
            try:
                out_data.to_csv(os.path.join(out_dir, subj+'_'+file_name+'_atlas-'+atlas_name+'.tsv'), sep='\t', index=False)
            except:
                os.makedirs(os.path.join(out_dir))
                out_data.to_csv(os.path.join(out_dir, subj+'_'+file_name+'_atlas-'+atlas_name+'.tsv'), sep='\t', index=False)
            print('done with file: \n %s.\n Input shape = %s\n Output shape = %s\n' % (file, subj_data.shape, out_data.shape))

def integrate_roi2BIDs(subj_list, event_template, roi_template, out_dir, out_name, TR):
    '''
    Extract data from roi files (generated by extract_rois_from_atlas), averaging across duraton following onset.
    Data is then combined into the BIDS event files and turned into one .csv output.

    subj_list [list of strings]
        e.g. ['sub-03', 'sub-04']
    event_template [string]
        Can include wildcards to use with glob. Can reference all event files (for all subjects), then the subject ID will be used to narrow the glob search.
    roi_templates [string,]
        glob template to find roi files from extract_rois_from_atlas.
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


    for subj in subj_list:
        print('working on subject: %s' % subj)
        roi_files = sorted(get_files(subj, roi_template)) # get sorted list of roi files.
        for rfile in roi_files: # append all rois to each other
            print('grabbing roi: %s' % rfile)
            rtemp = pd.read_csv(rfile, sep='\t', index_col=None)
            runlen_TR = len(rtemp.index) # get number of rows to get TR per run.
            if rfile == roi_files[0]: # append all run data as new rows.
                rdata = rtemp.copy()
            else:
                rdata = rdata.append(rtemp, ignore_index=True)

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
        r_avg = pd.DataFrame(np.empty((len(roi_locs), len(rdata.columns.values)), dtype=np.float),
                             columns = rdata.columns.values)
        r_avg[:] = np.nan
        # compute average for each roi
        for roi in rdata.columns.values:
            print('averaging roi: %s' % roi)
            r_avg[roi] = [np.mean(rdata[roi][r_locs]) for r_locs in roi_locs]
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


def extract_timecourse(subj, gm_file, func_file, out_dir, roi_path, out_label, check_output=None, dilate_roi=None, gm_method='scale', gm_thresh=None, export_nii=None, func_step=10):
    '''
    Extract ROI timecourse from functional data, given a list of nifti files.

    Works with 3d or 4d files, so can be applied to a timecouse, or to modeling output.

    Does not remove confounds. For that use, nilearn.img.clean_img

    [Required]
    subj = string, subject identifier, e.g. sub-001.
        Can also use some other tag here, e.g. 'stress_lvl2_speech-prep'
    gm_file = string, full path to gm mask, in same space as functional data.
        Be sure to include hemisphere information in the name.
        e.g. 'PATH/sub-001_fmriprep_skullstrip_ref_img.nii.gz__lh_REL.nii'
    func_file = string, full path to preprocessed functional data.
        e.g. PATH/'sub-001_task-rest_run-01_bold.nii.gz'
    out_dir = string, full path to folder to save outputs.
        e.g. '/home/project/outputs'
    roi_path = string, glob path to grab all ROI files.
        e.g. os.path.join(roi_dir, '*dil_ribbon_EPI_bin_ribbon.nii.gz')
    out_label = string, to be added to output files to specify anything you want,
        e.g. wm_Glasser

    [Optional]
    check_output [default = None] = set to True to print GM masked functional data and dilated ROIs.
    dilate_roi [default = None] = set to an Integer to dilate each ROI by X voxels.
    gm_method [default = 'scale'] Enter 'scale', 'above', or 'below'.
            'scale' gives a weighted average by multiplying the functional data by the GM mask.
            'between' incldues all voxels ABOVE the first and BELOW/EQUAL to the second value (given as a list in gm_thresh)
    gm_thresh [default = None] provide a float or list to use with thresholding in gm_method
        (e.g. [.2, 1.] to grab all gm voxels > .2)
    export_nii [default = None] = set to True to output a .nii file with ROI averages.
    func_step [default = 1] = Integer, denoting how many TRs to grab from functional data at once.
            Use this to optimize to memory availability. e.g. 10 works on my desktop. The server can most likely run 50
    '''
    import os, glob
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from nilearn.image import resample_img
    from scipy.ndimage.morphology import binary_dilation

    print('subj: ', subj)
    print('gm_file: ', gm_file)
    print('func_file:', func_file)
    print('out_dir:', out_dir)
    print('roi_path:', roi_path)
    print('out_label:', out_label)
    print('check_output:', check_output)
    print('dilate_roi:', dilate_roi)
    print('gm_method:', gm_method)
    print('gm_thresh:', gm_thresh)
    print('export_nii:', export_nii)
    print('func_step:', func_step, '\n\n')

    func_img = nib.load(func_file)
    print('linear neightbor interpolation of GM mask to functional space')
    fit_gm = resample_img(nib.load(gm_file),
                           target_affine=func_img.affine,
                           target_shape=func_img.shape[0:3],
                           interpolation='linear')

    # Set up a 4d array, to house all ROIs.
    roi_shape = [i for i in nib.load(func_file).shape[0:3]]
    roi_shape.append(len(glob.glob(roi_path)))
    roi_all = np.zeros(roi_shape)

    # loop through rois, reshaping each to functional space, and dilating if necessary.
    for idx, roi in enumerate(glob.glob(roi_path)):
        print('loading:', roi)
        print('nearest neightbor interpolation of ROI to functional space')
        fit_roi = resample_img(nib.load(roi),
                               target_affine=nib.load(func_file).affine,
                               target_shape=nib.load(func_file).shape[0:3],
                               interpolation='nearest')
        if dilate_roi:
            print('dilate ROI by', dilate_roi, 'voxels')
            fit_roi = nib.Nifti1Image(binary_dilation(fit_roi.get_fdata(), iterations=dilate_roi).astype(fit_roi.get_fdata().dtype),
                    fit_roi.affine, fit_roi.header)

        roi_all[...,idx] = fit_roi.get_fdata()

    if check_output: # save ROI, in case you want to check output.
        nib.save(nib.Nifti1Image(roi_all, fit_roi.affine, fit_roi.header),
                 os.path.join(out_dir, out_label+'_'+subj+'_all_roi.nii.gz'))

    # adjust length of slice loop, depending on whether image is 3d/4d
    try: # try to use the 4th dimension, will fail if there is none.
        TR_list=[*range(0, func_img.shape[3], func_step)]
        TR_len = func_img.shape[3]
    except:
        TR_list=[1] # set to 1 if no 4th dimension, then setup 3d image as 4d.
        TR_len = 1
        func_dat = func_img.get_fdata()
        func_dat = func_dat[...,None]
        assert func_step==1, 'If using a 3d functional image, func_step must be set at 1'

    for TR in range(0, TR_len, func_step):
        if len(func_img.shape)>3: # trigger on 4d images.
            if TR==TR_list[-1]:
                func_dat = func_img.dataobj[..., TR:] # grab slices from TR to end
                print('working on functional slices:', TR, 'to', TR_len)
            else:
                func_dat = func_img.dataobj[..., TR:TR+func_step] # TR sets lower bound
                print('working on functional slices:', TR, 'to', TR+func_step)

        print('apply GM mask to functional data.')
        if gm_method == 'scale':
            func_dat = func_dat*fit_gm.get_fdata()[...,None]
        elif gm_method == 'between':
            assert isinstance(gm_thresh, list), 'for gm_method=between, use a list of two float cutoff points for gm_thresh'
            func_dat = func_dat*np.where((fit_gm.get_fdata() > gm_thresh[0]) & (fit_gm.get_fdata() <= gm_thresh[1]), 1, np.nan)[...,None]

        for idx, roi in enumerate(glob.glob(roi_path)):
            print('extract roi:', roi)
            roi_dat = func_dat*roi_all[...,idx][...,None] # probabalistic mask.
            roi_flat = roi_dat[roi_all[...,idx]>0] # >0 to accomodate determiniatic and prob. masks.

            print('saving ROI average')
            if idx == 0:# if first ROI.
                TR_mean = np.nanmean(roi_flat, axis=0)
                TR_N = np.sum(~np.isnan(roi_flat[:,0]))
                TR_median_x = np.median(np.where(roi_all[...,idx]>0)[0])
                TR_median_y = np.median(np.where(roi_all[...,idx]>0)[1])
                TR_median_z = np.median(np.where(roi_all[...,idx]>0)[2])
            else:
                TR_mean = np.vstack((TR_mean, np.nanmean(roi_flat, axis=0))) # ROI x TRs
                TR_N = np.hstack((TR_N, np.sum(~np.isnan(roi_flat[:,0]))))
                TR_median_x = np.hstack((TR_median_x, np.median(np.where(roi_all[...,idx]>0)[0])))
                TR_median_y = np.hstack((TR_median_y, np.median(np.where(roi_all[...,idx]>0)[1])))
                TR_median_z = np.hstack((TR_median_z, np.median(np.where(roi_all[...,idx]>0)[2])))

        if TR == 0:# if first set of TRs.
            out_mean = TR_mean
            out_N = TR_N
            out_median_x = TR_median_x
            out_median_y = TR_median_y
            out_median_z = TR_median_z
        else:
            out_mean = np.hstack((out_mean, TR_mean)) # combine [ROI x TR] with [ROI x TR]

    # export data to .csv
    pd_out = pd.DataFrame({'subj':np.repeat(subj, len(glob.glob(roi_path))),
                           'tag':np.repeat(out_label, len(glob.glob(roi_path))),
                            'roi':[f.split('/')[-1] for f in glob.glob(roi_path)],
                            'cat_N':out_N,
                            'median_x':out_median_x,
                            'median_y':out_median_y,
                            'median_z':out_median_z})
    if len(TR_list) > 1: # no STD on 3d images.
        pd_out['roi_sd'] = np.nanstd(out_mean, axis=1)
    pd_out = pd_out.join(pd.DataFrame(out_mean.reshape(len(out_mean),-1)).add_suffix('_mean'))
    pd_out.to_csv(os.path.join(out_dir,
                               os.path.join(out_dir, out_label+'_'+subj+'_'+func_file.split('/')[-1].split('.nii.gz')[0]+'.csv')),
                  index=False, header=True)

    if export_nii: # export nifti
        nii_mean = np.zeros(nib.load(func_file).shape[0:3])
        print('writing nifti for ROI means.')
        for idx, roi in enumerate(glob.glob(roi_path)):
            nii_mean[roi_all[...,idx]>0] = np.nanmean(out_mean[idx,:]) # This is mean of voxel means across TRs.
        nii_mean_nib = nib.Nifti1Image(nii_mean, nib.load(func_file).affine, nib.load(func_file).header)
        nii_mean_nib.header['cal_max'] = np.nanmax(nii_mean) # adjust min and max header info.
        nii_mean_nib.header['cal_min'] = np.nanmin(nii_mean)
        nib.save(nii_mean_nib,
                 os.path.join(out_dir, out_label+'_'+subj+'_mean_'+func_file.split('/')[-1]))
        if len(TR_list) > 1:
            nii_sd = np.zeros(nib.load(func_file).shape[0:3])
            nii_sd[roi_all[...,idx]>0] = np.nanstd(out_mean[idx,:]) # This is mean of voxel means across TRs.
            nii_sd_nib = nib.Nifti1Image(nii_sd, nib.load(func_file).affine, nib.load(func_file).header)
            nii_sd_nib.header['cal_max'] = np.nanmax(nii_sd) # adjust min and max header info.
            nii_sd_nib.header['cal_min'] = np.nanmin(nii_sd)
            nib.save(nii_sd_nib,
                     os.path.join(out_dir, out_label+'_'+subj+'_sd_'+func_file.split('/')[-1]))

    print('####\ndone with %s \n####' % subj)

def extract_voxels(subj, gm_file, func_file, out_dir, roi_path, out_label, export_voxels, dilate_roi=None, func_step=10):
    '''
    Extract voxels from ROIs in functional data, given a list of nifti files.
    Works with 3d or 4d files, so can be applied to a timecouse, or to modeling output.
    Does not remove confounds. For that use, nilearn.img.clean_img

    [Required]
    subj = string, subject identifier, e.g. sub-001.
        Can also use some other tag here, e.g. 'stress_lvl2_speech-prep'
    gm_file = string, full path to gm mask, in same space as functional data.
        Be sure to include hemisphere information in the name.
        e.g. 'PATH/sub-001_fmriprep_skullstrip_ref_img.nii.gz__lh_REL.nii'
    func_file = string, full path to preprocessed functional data.
        e.g. PATH/'sub-001_task-rest_run-01_bold.nii.gz'
    out_dir = string, full path to folder to save outputs.
        e.g. '/home/project/outputs'
    roi_path = string, glob path to grab all ROI files.
        e.g. os.path.join(roi_dir, '*dil_ribbon_EPI_bin_ribbon.nii.gz')
    out_label = string, to be added to output files to specify anything you want,
        e.g. wm_Glasser
    export_voxels = list of strings to extract voxels from those ROIs containing each string.
                e.g. ['Caudate', 'Amygdala']

    [Optional]
    dilate_roi [default = None] = set to an Integer to dilate each ROI by X voxels.
    func_step [default = 1] = Integer, denoting how many TRs to grab from functional data at once.
            Use this to optimize to memory availability. e.g. 10 works on my desktop. The server can most likely run 50
    '''
    import os, glob
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from nilearn.image import resample_img
    from scipy.ndimage.morphology import binary_dilation

    print('subj: ', subj)
    print('gm_file: ', gm_file)
    print('func_file:', func_file)
    print('out_dir:', out_dir)
    print('roi_path:', roi_path)
    print('out_label:', out_label)
    print('check_output:', check_output)
    print('dilate_roi:', dilate_roi)
    print('export_sd:', export_sd)
    print('export_voxels:', export_voxels)
    print('func_step:', func_step, '\n\n')

    func_img = nib.load(func_file)
    print('linear neightbor interpolation of GM mask to functional space')
    fit_gm = resample_img(nib.load(gm_file),
                           target_affine=func_img.affine,
                           target_shape=func_img.shape[0:3],
                           interpolation='linear')

    # adjust length of slice loop, depending on whether image is 3d/4d
    try: # try to use the 4th dimension, will fail if there is none.
        TR_list=[*range(0, func_img.shape[3], func_step)]
        TR_len = func_img.shape[3]
    except:
        TR_list = [1]
        TR_len = 1
        func_dat = func_img.get_fdata()
        func_dat = func_dat[...,None]
        assert func_step==1, 'If using a 3d functional image, func_step must be set at 1'

    # resample ROI to subject space.
    for roi in glob.glob(roi_path):
        if any(r in roi for r in export_voxels):
            print('working on:', roi)
            print('nearest neightbor interpolation of ROI to functional space')
            fit_roi = resample_img(nib.load(roi),
                                   target_affine=nib.load(func_file).affine,
                                   target_shape=nib.load(func_file).shape[0:3],
                                   interpolation='nearest')
            if dilate_roi:
                print('dilate ROI by', dilate_roi, 'voxels. \n WARNING: THIS WILL REMOVE ANY PROBABLISTIC MAPPING AND SWITCH TO BINARY')
                fit_roi = nib.Nifti1Image(binary_dilation(fit_roi.get_fdata(), iterations=dilate_roi).astype(fit_roi.get_fdata().dtype),
                        fit_roi.affine, fit_roi.header)
            if check_output:
                nib.save(fit_roi, os.path.join(out_dir, out_label+'_'+subj+'_'+roi.split('/')[-1]))

            for idx, TR in enumerate(range(0, TR_len, func_step)):
                if len(func_img.shape)>3: # if 4d file
                    if TR==TR_list[-1]:
                        func_dat = func_img.dataobj[..., TR:] # grab slices from TR to end
                        print('working on functional slices:', TR, 'to', TR_len)
                    else:
                        func_dat = func_img.dataobj[..., TR:TR+func_step] # TR sets lower bound
                        print('working on functional slices:', TR, 'to', TR+func_step)

                print('grab ROI voxels from functional data, then averaging')
                roi_dat = func_dat*fit_roi.get_fdata()[...,None] # probabalistic mask.
                roi_flat = roi_dat[fit_roi.get_fdata()>0] # >0 to accomodate determiniatic and prob. masks.
                if TR == 0: # stack voxels in ROI across all TRs.
                    roi_flat_all = roi_flat
                else:
                    roi_flat_all = np.hstack((roi_flat_all, roi_flat))

            # Output for ROI.
            pd_roi = pd.DataFrame({'subj':np.repeat(subj, roi_flat_all.shape[0]),
                                   'tag':np.repeat(out_label, roi_flat_all.shape[0]),
                                   'roi':np.repeat(roi.split('/')[-1], roi_flat_all.shape[0]),
                                   'roi_prob': fit_roi.get_fdata()[fit_roi.get_fdata()>0],
                                   'gm_prob':fit_gm.get_fdata()[fit_roi.get_fdata()>0],
                                   'x_loc':np.where(fit_roi.get_fdata()>0)[0],
                                   'y_loc':np.where(fit_roi.get_fdata()>0)[1],
                                   'z_loc':np.where(fit_roi.get_fdata()>0)[2]})
            pd_roi = pd_roi.join(pd.DataFrame(roi_flat_all.reshape(len(roi_flat_all), -1)))
            pd_roi.to_csv(
                os.path.join(out_dir,
                             out_label+'_'+subj+'_voxels_'+roi.split('/')[-1].split('.nii.gz')[0]+'.csv'),
            index=False, header=True)

    print('####\ndone with %s \n####' % subj)
