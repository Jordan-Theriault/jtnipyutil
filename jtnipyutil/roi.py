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
