def create_lvl2tfce_wf(mask=False):
    '''
    Input [Mandatory]:

        ~~~~~~~~~~~ Set through inputs.inputspec

        fwhm_list: list of strings representing smoothing kernels. Can be run iterably.
            'None' represents no smoothing.
            e.g. ['none', '1.5', '6']
            ** Often you will want to input this with an iterable node.

        contrast: Character defining contrast name.
            Name should match a dictionary entry in full_cons and con_regressors.
            ** Often you will want to input this with an iterable node.

        full_cons: dictionary of each contrast.
            Names should match con_regressors.
            Entries in format [('name', 'stat', [condition_list], [weight])]
            e.g. full_cons = {
                '1_instructions_Instructions': [('1_instructions_Instructions', 'T', ['1_instructions_Instructions'], [1])]
                }

        input_dir: string, representing directory to level1 data, modeled using jt_modeling.create_lvl1pipe_wf.
            e.g. inputs.inputspec.input_dir = '/home/neuro/data'

        output_dir: string, representing directory of output.
            e.g. inputs.inputspec.output_dir ='/home/neuro/output'

        subject_list: list of string, with BIDs-format IDs to identify subjects.
            Use this to drop high movement subjects, even if they are among other files that will be grabbed.
            e.g. inputs.inputspec.subject_list =['sub-001', sub-002']

        con_regressors: dictionary of by-subject regressors for each contrast.
                Names should match full_cons.
                e.g. inputs.inputspec.con_regressors = {
                        '1_instructions_Instructions': {'1_instructions_Instructions': [1] * len(subject_list),
                        'reg2': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        'reg3': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                        }
                    }
        Input [Optional]:
            mask: [default: False] path to mask file. Can have different dimensions from functional data, but should obviously be in the same reference space if anatomical (see jt_util.align_mask).
                e.g. inputs.inputspec.mask_file = '/home/neuro/atlases/FSMAP/stress/realigned_masks/amygdala_bl_flirt.nii.gz'

            sinker_subs: list of tuples, each containing a pair of strings.
                These will be sinker substitutions. They will change filenames in the output folder.
                Usually best to run the pipeline once, before deciding on these.
                e.g. inputs.inputspec.sinker_subs = [('tstat', 'raw_tstat'),
                       ('tfce_corrp_raw_tstat', 'tfce_corrected_p')]

        Output:
            lvl2tfce_wf: workflow to perform second-level modeling, using threshold free cluster estimation (tfce; see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise/UserGuide)

    '''
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function
    ##################  Setup workflow.
    lvl2tfce_wf = pe.Workflow(name='lvl2tfce_wf')

    inputspec = pe.Node(IdentityInterface(
        fields=['input_dir',
                'output_dir',
                'mask_file',
                'subject_list',
                'con_regressors',
                'full_cons',
                'sinker_subs',
                'fwhm',
                'contrast'
                ],
        mandatory_inputs=False),
                 name='inputspec')
    if mask:
        inputspec.inputs.mask_file = mask

    ################## Make template
    def mk_outdir(output_dir, mask=False):
        import os
        from time import gmtime, strftime
        time_prefix = strftime("%Y-%m-%d_%Hh-%Mm", gmtime())+'_'
        if mask:
            new_out_dir = os.path.join(output_dir, time_prefix + mask.split('/')[-1].split('.')[0])
        else:
            new_out_dir = os.path.join(output_dir, time_prefix + 'wholebrain')
        if not os.path.isdir(new_out_dir):
            os.makedirs(new_out_dir)
        return new_out_dir

    make_outdir = pe.Node(Function(input_names=['output_dir', 'mask'],
                                   output_names=['new_out_dir'],
                                   function=mk_outdir),
                          name='make_outdir')
    # make_template.inputs.output_dir = from inputspec
    # make_template.inputs.mask = from inputspec

    def get_template(fwhm, contrast, input_dir, output_dir):
        import os
        # makes template to grab copes files, based on the smoothing kernel being processed.
        if fwhm == 'none':
            con_file = 'cope'+contrast+'.nii.gz'
            template={
                'cope': os.path.join(input_dir, 'nosmooth', 'sub-*/model/*/*', con_file)
            }
            out_path = os.path.join(output_dir, 'nosmooth')
        else:
            fwhm_path = 'fwhm_'+fwhm
            con_file = 'cope'+contrast+'.nii.gz'
            template={
                'cope': os.path.join(input_dir, 'smooth', 'sub-*/model',
                                     ['*',fwhm_path, '*'], '*', con_file)
            }
            out_path = os.path.join(output_dir, fwhm)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        return template, out_path

    make_template = pe.Node(Function(input_names=['fwhm', 'contrast', 'input_dir', 'output_dir'],
                                     output_names=['template', 'out_path'],
                                     function=get_template),
                            name='make_template')
    # make_template.inputs.output_dir = from make_outdir
    # make_template.inputs.input_dir = from inputspec
    # make_template.inputs.fwhm # From inputspec.

    ################## Get contrast
    def get_con(contrast, full_cons, con_regressors):
        con_info = full_cons[contrast]
        reg_info = con_regressors[contrast]
        return con_info, reg_info

    get_model_info = pe.Node(Function(input_names=['contrast', 'full_cons', 'con_regressors'],
                                      output_names=['con_info', 'reg_info'],
                                      function=get_con),
                             name='get_model_info')
    # get_model_info.inputs.full_cons = From inputspec
    # get_model_info.inputs.full_regs = From inputspec

    ################## Get files
    def get_files(subject_list, template):
        import glob
        temp_list = []
        out_list = []
        for x in glob.glob(list(template.values())[0]):
            if any(subj in x for subj in subject_list):
                temp_list.append(x)
        for file in temp_list: # ensure no duplicate entries.
            if file not in out_list:
                out_list.append(file)
        return out_list

    get_copes = pe.Node(Function(
        input_names=['subject_list', 'template'],
        output_names=['out_list'],
        function=get_files),
                        name='get_copes')
    # get_copes.inputs.subject_list = # From inputspec
    # get_copes.inputs.template = template # From make_template.

    ################## Merge into 4d files.
    import nipype.interfaces.fsl as fsl # fsl
    merge_copes = pe.Node(interface=fsl.Merge(dimension='t'),
                    name='merge_copes')
    # merge_copes.inputs.in_files = copes

    ################## Level 2 design.
    level2model = pe.Node(interface=fsl.MultipleRegressDesign(),
                        name='level2model')
    # level2model.inputs.contrasts # from get_con_info
    # level2model.inputs.regressors # from get_con_info

    ################## Fit mask, if given 2 design.
    if mask:
        from jtnipyutil.util import fit_mask
        fit_mask = pe.Node(Function(
            input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'],
            function=fit_mask),
                            name='fit_mask')

    ################## FSL Randomize.
    randomise = pe.Node(interface=fsl.Randomise(), name = 'randomise')
    # randomise.inputs.in_file = #From merge_copes
    # randomise.inputs.design_mat = # From level2model design_mat
    # randomise.inputs.tcon = # From level2model design_con
    # randomise.inputs.cm_thresh = 2.49 # mass based cluster thresholding. Not used.
    # randomise.mask = # Provided from mask_reslice, if mask provided.
    randomise.inputs.tfce = True
    randomise.inputs.raw_stats_imgs = True
    randomise.inputs.vox_p_values = True
    # randomise.inputs.num_perm = 5000

    def adj_minmax(in_file):
        import nibabel as nib
        import numpy as np
        import os
        img = nib.load(in_file[0])
        data = img.get_data()
        img.header['cal_max'] = np.max(data)
        img.header['cal_min'] = np.min(data)
        nib.save(img, in_file[0])
        return in_file

    ################## Setup datasink.
    from nipype.interfaces.io import DataSink
    import os
    # sinker = pe.Node(DataSink(parameterization=False), name='sinker')
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')

    ################## Setup Pipeline.
    lvl2tfce_wf.connect([
        (inputspec, make_outdir, [('output_dir', 'output_dir')]),
        (make_outdir, make_template, [('new_out_dir', 'output_dir')]),
        (inputspec, make_template, [('fwhm', 'fwhm'),
                                    ('contrast', 'contrast'),
                                    ('input_dir', 'input_dir')]),
        (inputspec, get_model_info, [('full_cons', 'full_cons'),
                                    ('con_regressors', 'con_regressors')]),
        (inputspec, get_model_info, [('contrast', 'contrast')]),
        (inputspec, get_copes, [('subject_list', 'subject_list')]),
        (make_template, get_copes, [('template', 'template')]),
        (get_copes, merge_copes, [('out_list', 'in_files')]),
        (get_model_info, level2model, [('con_info', 'contrasts')]),
        (get_model_info, level2model, [('reg_info', 'regressors')]),
        (merge_copes, randomise, [('merged_file', 'in_file')]),
        (level2model, randomise, [('design_mat', 'design_mat')]),
        (level2model, randomise, [('design_con', 'tcon')]),
        ])
    if mask:
        lvl2tfce_wf.connect([
            (inputspec, fit_mask, [('mask_file', 'mask_file')]),
            (merge_copes, fit_mask, [('merged_file', 'ref_file')]),
            (fit_mask, randomise, [('out_mask', 'mask')]),
            (inputspec, make_outdir, [('mask_file', 'mask')]),
            (fit_mask, sinker, [('out_mask', 'out.@mask')]),
            ])

    lvl2tfce_wf.connect([
        (inputspec, sinker, [('sinker_subs', 'substitutions')]),
        (make_template, sinker, [('out_path', 'base_directory')]),
        (level2model, sinker, [('design_con', 'out.@con')]),
        (level2model, sinker, [('design_grp', 'out.@grp')]),
        (level2model, sinker, [('design_mat', 'out.@mat')]),
        (randomise, sinker, [(('t_corrected_p_files', adj_minmax), 'out.@t_cor_p')]),
        (randomise, sinker, [(('tstat_files', adj_minmax), 'out.@t_stat')]),
        # (randomise, sinker, [('t_corrected_p_files', 'out.@t_cor_p')]),
        # (randomise, sinker, [('tstat_files', 'out.@t_stat')]),
        ])
    return lvl2tfce_wf





def create_lvl1pipe_wf(options):
    '''
    Input [Mandatory]:
        ~~~~~~~~~~~ Set in command call:

        options: dictionary with the following entries
            remove_steadystateoutlier [boolean]:
                Should always be True. Remove steady state outliers from bold timecourse, specified in fmriprep confounds file.
            smooth [boolean]:
                If True, then /smooth subfolder created and populated with results. If False, then /nosmooth subfolder created and populated with results.
            censoring [string]:
                Either '' or 'despike', which implements nipype.interfaces.afni.Despike
            ICA_AROMA [boolean]:
                Use AROMA error components, from fmriprep confounds file.
            run_contrasts [boolean]:
                If False, then components related to contrasts and p values are removed from   nipype.workflows.fmri.fsl.estimate.create_modelfit_workflow()
            keep_resid [boolean]:
                If False, then only sum of squares residuals will be outputted. If True, then timecourse residuals kept.

        ~~~~~~~~~~~ Set through inputs.inputspec

        input_dir [string]:
            path to folder containing fmriprep preprocessed data.
            e.g. model_wf.inputs.inputspec.input_dir = '/home/neuro/data'
        output_dir [string]:
            path to desired output folder. Workflow will create a new subfolder based on proj_name.
            e.g. model_wf.inputs.inputspec.output_dir = '/home/neuro/output'
        proj_name [string]:
            name for project subfolder within output_dir. Ideally something unique, or else workflow will write to an existing folder.
            e.g. model_wf.inputs.inputspec.proj_name = 'FSMAP_stress'
        design_col [string]:
            Name of column within events.tsv with values corresponding to entries specified in params.
            e.g. model_wf.inputs.inputspec.design_col = 'trial_type'
        params [list fo strings]:
            values within events.tsv design_col that correspond to events to be modeled.
            e.g. model_wf.inputs.inputspec.params = ['Instructions', 'Speech_prep', 'No_speech']
        contrasts [list of lists]:
            Specifies contrasts to be performed. using params selected above.
            e.g. model_wf.inputs.inputspec.contrasts =
                [['Instructions', 'T', ['Instructions'], [1]],
                 ['Speech_prep', 'T', ['Speech_prep'], [1]],
                 ['No_speech', 'T', ['No_speech'], [1]],
                 ['Speech_prep>No_speech', 'T', ['Speech_prep', 'No_speech'], [1, -1]]]
        noise_regressors [list of strings]:
            column names in confounds.tsv, specifying desired noise regressors for model.
            IF noise_transforms are to be applied to a regressor, add '*' to the name.
            e.g. model_wf.inputs.inputspec.noise_regressors = ['CSF', 'WhiteMatter', 'GlobalSignal', 'X*', 'Y*', 'Z*', 'RotX*', 'RotY*', 'RotZ*']
        noise_transforms [list of strings]:
            noise transforms to be applied to select noise_regressors above. Possible values are 'quad', 'tderiv', and 'quadtderiv', standing for quadratic function of value, temporal derivative of value, and quadratic function of temporal derivative.
            e.g. model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
        TR [float]:
            Scanner TR value in seconds.
            e.g. model_wf.inputs.inputspec.TR = 2.
        FILM_threshold [integer]:
            Cutoff value for modeling threshold. 1000: p <.001; 1: p <=1, i.e. unthresholded.
            e.g. model_wf.inputs.inputspec.FILM_threshold = 1
        hpf_cutoff [float]:
            high pass filter value.
            e.g. model_wf.inputs.inputspec.hpf_cutoff = 120.
        bases: (a dictionary with keys which are 'hrf' or 'fourier' or 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
             dict {'name':{'basesparam1':val,...}}
             name : string
             Name of basis function (hrf, fourier, fourier_han, gamma, fir)
             hrf :
                 derivs : 2-element list
                    Model HRF Derivatives. No derivatives: [0,0],
                    Time derivatives : [1,0],
                    Time and Dispersion derivatives: [1,1]
             fourier, fourier_han, gamma, fir:
                 length : int
                    Post-stimulus window length (in seconds)
                 order : int
                    Number of basis functions
            e.g. model_wf.inputs.inputspec.bases = {'dgamma':{'derivs': False}}
        model_serial_correlations [boolean]:
            Allow prewhitening, with 5mm spatial smoothing.
            model_wf.inputs.inputspec.model_serial_correlations = True
        sinker_subs [list of tuples]:
            passed to nipype.interfaces.io.Datasink. Changes names when passing to output directory.
            e.g. model_wf.inputs.inputspec.sinker_subs =
                [('pe1', 'pe1_instructions'),
                 ('pe2', 'pe2_speech_prep'),
                 ('pe3', 'pe3_no_speech')]
        bold_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant BOLD files. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.bold_template =
                {'bold': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'}
                 This would grab the functional run for all subjects, and when subject_id = 'sub-001', there is ONE file in the list that the ID could possible correspond to.
                To handle multiple runs, list the run information in the subject_id. e.g. 'sub-01_task-trag_run-01'.
        mask_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant MASK files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.mask_template =
            {'mask': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'}
            See bold_template for more detail.
        task_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant events.tsv files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.task_template =
            {'task': '/home/neuro/data/sub-*/func/sub-*_task-stress_events.tsv'}
            See bold_template for more detail.
        confound_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant confounds.tsv files, corresponding to functional images. Each subject_list entry should uniquely identify the ONE relevant file.
            e.g. model_wf.inputs.inputspec.confound_template =
            {'confound': '/home/neuro/data/sub-*/func/sub-*_task-stress_bold_confounds.tsv'}
            See bold_template for more detail.
        smooth_gm_mask_template [dictionary with string entry]:
            Specifies path, with wildcard, to grab all relevant grey matter mask .nii.gz files, pulling from each subject's /anat fodler. Each subject_list entry should uniquely identify the ONE relevant file (BUT SEE THE NOTE BELOW).
            e.g. model_wf.inputs.inputspec.smooth_gm_mask_template =
                {'gm_mask': '/scratch/data/sub-*/anat/sub-*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz'}
                NOTE: If the subject_id value has more information than just the ID (e.g. sub-01_task-trag_run-01), then JUST the sub-01 portion will be used to identify the grey matter mask. This is because multiple runs will have the same anatomical data. i.e. sub-01_run-01, sub-01_run-02, sub-01_run-03, all correspond to sub-01_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz.
        fwhm [float]. Redundant if options['smooth']: False
            Determines smoothing kernel. Multiple kernels can be run in parallel by iterating through an outside workflow. Also see subject_id below for another example of iterables.
            e.g.
                model_wf.inputs.inputspec.fwhm = 1.5
            OR Iterable e.g.
                import nipype.pipeline.engine as pe
                fwhm_list = [1.5, 6]
                infosource = pe.Node(IdentityInterface(fields=['fwhm']),
                           name='infosource')
                infosource.iterables = [('fwhm', fwhm_list)]
                full_model_wf = pe.Workflow(name='full_model_wf')
                full_model_wf.connect([(infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])
                full_model_wf.run()
        subject_id [string]:
            Identifies subject in conjnuction with template. See bold_template note above.
            Can also be entered as an iterable from an outside workflow, in which case iterables are run in parallel to the extent that cpu cores are available.
            e.g.
                model_wf.inputs.inputspec.subject_id = 'sub-01'
            OR Iterable e.g.
                import nipype.pipeline.engine as pe
                subject_list = ['sub-001', 'sub-002']
                infosource = pe.Node(IdentityInterface(fields=['subject_id']),
                           name='infosource')
                infosource.iterables = [('subject_id', subject_list)]
                full_model_wf = pe.Workflow(name='full_model_wf')
                full_model_wf.connect([(infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])
                full_model_wf.run()
    '''
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface, SelectFiles
    from nipype.interfaces.utility.wrappers import Function

    ##################  Setup workflow.
    lvl1pipe_wf = pe.Workflow(name='lvl_one_pipe')

    inputspec = pe.Node(IdentityInterface(
        fields=['input_dir',
                'output_dir',
                'design_col',
                'noise_regressors',
                'noise_transforms',
                'TR', # in seconds.
                'FILM_threshold',
                'hpf_cutoff',
                'params',
                'contrasts',
                'bases',
                'model_serial_correlations',
                'sinker_subs',
                'bold_template',
                'mask_template',
                'task_template',
                'confound_template',
                'smooth_gm_mask_template',
                'gmmask_args',
                'subject_id',
                'fwhm',
                'proj_name',
                ],
        mandatory_inputs=False),
                 name='inputspec')

    ################## Select Files
    def get_file(subj_id, template):
        import glob
        temp_list = []
        out_list = []
        if '_' in subj_id and '/anat/' in list(template.values())[0]:
            subj_id = subj_id[:subj_id.find('_')]
            # if looking for gmmask, and subj_id includes additional info (e.g. sub-001_task-trag_run-01) then just take the subject id component, as the run info will not be present for the anatomical data.
        for x in glob.glob(list(template.values())[0]):
            if subj_id in x:
                temp_list.append(x)
        for file in temp_list: # ensure no duplicate entries.
            if file not in out_list:
                out_list.append(file)
        assert (len(out_list) == 1), 'Each combination of template and subject ID should return 1 file. either one file, or multiple files returned.'
        out_file = out_list[0]
        return out_file

    get_bold = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_bold')
    get_mask = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_mask')
    get_task = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_task')
    get_confile = pe.Node(Function(
        input_names=['subj_id', 'template'],
        output_names=['out_file'],
        function=get_file),
                        name='get_confile')
    # get_bold.inputs.subj_id # From inputspec
    # get_bold.inputs.templates # From inputspec
    if options['smooth']:
        get_gmmask = pe.Node(Function(
            input_names=['subj_id', 'template'],
            output_names=['out_file'],
            function=get_file),
                            name='get_gmmask')

        mod_gmmask = pe.Node(fsl.maths.MathsCommand(),
                                name='mod_gmmask')
        # mod_gmmask.inputs.in_file = # from get_gmmask
        # mod_gmmask.inputs.args = from inputspec
        from jtnipyutil.util import fit_mask
        fit_mask = pe.Node(Function(
            input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'],
            function=fit_mask),
                            name='fit_mask')

    ################## Setup confounds
    def get_terms(confound_file, noise_transforms, noise_regressors, TR, options):
        # # Add time derivs, quadratic terms, and quad time derives if requested.
        import numpy as np
        import pandas as pd
        df_cf = pd.DataFrame(pd.read_csv(confound_file, sep='\t', parse_dates=False))
        if len(df_cf.columns[df_cf.columns.to_series().str.contains('_r')]) > 0: # if multiple runs were merged then fix the noise_regresor list.
            noise_regressors_r = []
            for run in range(1, 1000):
                run_sufx = '_r' + str(run)
                if len(df_cf.columns[df_cf.columns.to_series().str.contains(run_sufx)]) == 0: # if no run for current loop found, break and proceed.
                    break
                noise_regressors_r.append([s + run_sufx for s in noise_regressors])
            noise_regressors = []
            for sublist in noise_regressors_r:
                for item in sublist:
                    noise_regressors.append(item)
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
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]])
        if options['ICA_AROMA']:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]])
        if len(df_cf.columns[df_cf.columns.to_series().str.contains('run_')]) > 0: # get runs if there are any, assuming combine_runs was used.
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('run_')]])
        return confounds

    get_confounds = pe.Node(Function(input_names=['confound_file', 'noise_transforms',
                                                  'noise_regressors', 'TR', 'options'],
                                 output_names=['confounds'],
                                  function=get_terms),
                         name='get_confounds')
    # get_confounds.inputs.confound_file =  # From get_confile
    # get_confounds.inputs.noise_transforms =  # From inputspec
    # get_confounds.inputs.noise_regressors =  # From inputspec
    # get_confounds.inputs.TR =  # From inputspec
    get_confounds.inputs.options = options

    ################## Create bunch to run FSL first level model.
    def get_subj_info(task_file, design_col, confounds, params):
        # Makes a Bunch, giving all necessary data about conditions, onsets, and durations to
        # FSL first level model. Needs a task file to run.
        from nipype.interfaces.base import Bunch
        import pandas as pd
        import numpy as np
        output = []
        tf = task_file
        df = pd.DataFrame(pd.read_csv(tf, sep='\t', parse_dates=False))
        output = Bunch(conditions= params,
                           onsets=[list(df[df[design_col] == f].onset) for f in params],
                           durations=[list(set(df[df[design_col] == f].duration)) for f in params],
                           amplitudes=None,
                           tmod=None,
                           pmod=None,
                           regressor_names=confounds.columns.values,
                           regressors=confounds.T.values.tolist()) # movement regressors added here. List of lists.
        return output

    make_bunch = pe.Node(Function(input_names=['task_file', 'design_col', 'confounds', 'params'],
                                 output_names=['subject_info'],
                                  function=get_subj_info),
                         name='make_bunch')
    # make_bunch.inputs.task_file =  # From get_task
    # make_bunch.inputs.confounds =  # From get_confounds
    # make_bunch.inputs.design_col =  # From inputspec
    # make_bunch.inputs.params =  # From inputspec

    def mk_outdir(output_dir, options, proj_name):
        import os
        from time import gmtime, strftime
        prefix = proj_name
        if options['smooth']:
            new_out_dir = os.path.join(output_dir, prefix, 'smooth')
        else:
            new_out_dir = os.path.join(output_dir, prefix, 'nosmooth')
        if not os.path.isdir(new_out_dir):
            os.makedirs(new_out_dir)
        return new_out_dir

    make_outdir = pe.Node(Function(input_names=['output_dir', 'options', 'proj_name'],
                                   output_names=['new_out_dir'],
                                   function=mk_outdir),
                          name='make_outdir')
    # make_outdir.inputs.proj_name = from inputspec
    # make_outdir.inputs.output_dir = from inputspec
    make_outdir.inputs.options = options


    ################## Mask functional data.
    from jtnipyutil.util import mask_img
    maskBold = pe.Node(Function(input_names=['img_file', 'mask_file'],
                                output_names=['out_file'],
                                function=mask_img),
                      name='maskBold')
    # maskBold.inputs.img_file # From get_bold, or smooth_wf
    # maskBold.inputs.mask_file # From get_mask

    ################## Despike
    from nipype.interfaces.afni import Despike
    despike = pe.Node(Despike(),
                      name='despike')
    # despike.inputs.in_file = # From Mask
    despike.inputs.outputtype = 'NIFTI_GZ'

    from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
    smooth_wf = create_susan_smooth()
    # smooth_wf.inputs.inputnode.in_files = # from maskBold
    # smooth_wf.inputs.inputnode.fwhm = # from inputspec

    ################## Model Generation.
    import nipype.algorithms.modelgen as model
    specify_model = pe.Node(interface=model.SpecifyModel(), name='specify_model')
    specify_model.inputs.input_units = 'secs'
    # specify_model.functional_runs # From maskBold, despike, or smooth_wf
    # specify_model.subject_info # From subject_info
    # specify_model.high_pass_filter_cutoff # From inputspec
    # specify_model.time_repetition # From inputspec

    ################## Estimate workflow
    from nipype.workflows.fmri.fsl import estimate # fsl workflow
    modelfit = estimate.create_modelfit_workflow()
    modelfit.base_dir = '.'
    # modelfit.inputs.inputspec.session_info = # From specify_model
    # modelfit.inputs.inputspec.functional_data = # from maskBold
    # modelfit.inputs.inputspec.interscan_interval = # From inputspec
    # modelfit.inputs.inputspec.film_threshold = # From inputspec
    # modelfit.inputs.inputspec.bases = # From inputspec
    # modelfit.inputs.inputspec.model_serial_correlations = # From inputspec
    # modelfit.inputs.inputspec.contrasts = # From inputspec

    if not options['run_contrasts']: # drop contrast part of modelfit if contrasts aren't required.
        modelestimate = modelfit.get_node('modelestimate')
        merge_contrasts = modelfit.get_node('merge_contrasts')
        ztop = modelfit.get_node('ztop')
        outputspec = modelfit.get_node('outputspec')
        modelfit.disconnect([(modelestimate, merge_contrasts, [('zstats', 'in1'),
                                                             ('zfstats', 'in2')]),
                             (merge_contrasts, ztop, [('out', 'in_file')]),
                             (merge_contrasts, outputspec, [('out', 'zfiles')]),
                             (ztop, outputspec, [('out_file', 'pfiles')])
                            ])
        modelfit.remove_nodes([merge_contrasts, ztop])

    ################## DataSink
    from nipype.interfaces.io import DataSink
    import os.path
    sinker = pe.Node(DataSink(), name='sinker')
    # sinker.inputs.substitutions = # From inputspec
    # sinker.inputs.base_directory = # frm make_outdir

    def negate(input):
        return not input

    def unlist(input):
        return input[0]

    lvl1pipe_wf.connect([
        # grab subject/run info
        (inputspec, get_bold, [('subject_id', 'subj_id'),
                                ('bold_template', 'template')]),
        (inputspec, get_mask, [('subject_id', 'subj_id'),
                                ('mask_template', 'template')]),
        (inputspec, get_task, [('subject_id', 'subj_id'),
                                ('task_template', 'template')]),
        (inputspec, get_confile, [('subject_id', 'subj_id'),
                                ('confound_template', 'template')]),
        (inputspec, get_confounds, [('noise_transforms', 'noise_transforms'),
                                     ('noise_regressors', 'noise_regressors'),
                                     ('TR', 'TR')]),
        (inputspec, make_bunch, [('design_col', 'design_col'),
                                  ('params', 'params')]),
        (inputspec, make_outdir, [('output_dir', 'output_dir'),
                                  ('proj_name', 'proj_name')]),
        (inputspec, specify_model, [('hpf_cutoff', 'high_pass_filter_cutoff'),
                                     ('TR', 'time_repetition')]),
        (inputspec, modelfit, [('TR', 'inputspec.interscan_interval'),
                                ('FILM_threshold', 'inputspec.film_threshold'),
                                ('bases', 'inputspec.bases'),
                                ('model_serial_correlations', 'inputspec.model_serial_correlations'),
                                (('model_serial_correlations', negate), 'modelestimate.autocorr_noestimate'),
                                ('contrasts', 'inputspec.contrasts')]),
        (get_confile, get_confounds, [('out_file', 'confound_file')]),
        (get_confounds, make_bunch, [('confounds', 'confounds')]),
        (get_task, make_bunch, [('out_file', 'task_file')]),
        (make_bunch, specify_model, [('subject_info', 'subject_info')]),
        (get_mask, maskBold, [('out_file', 'mask_file')]),
        ])

    if options['censoring'] == 'despike':
        lvl1pipe_wf.connect([
            (get_bold, despike, [('out_file', 'in_file')])
            ])
        if options['smooth']:
            lvl1pipe_wf.connect([
                (inputspec, smooth_wf, [('fwhm', 'inputnode.fwhm')]),
                (inputspec, get_gmmask, [('subject_id', 'subj_id'),
                                        ('smooth_gm_mask_template', 'template')]),
                (get_gmmask, mod_gmmask, [('out_file', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (mod_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (fit_mask, smooth_wf, [('out_mask', 'inputnode.mask_file')]),
                (fit_mask, sinker, [('out_mask', 'smoothing_mask')]),
                (despike, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [(('outputnode.smoothed_files', unlist), 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
                ])
        else:
            lvl1pipe_wf.connect([
                (despike, specify_model, [('out_file', 'functional_runs')]),
                (despike, modelfit, [('out_file', 'inputspec.functional_data')]),
                (despike, sinker, [('out_file', 'despike')])
                ])
    else:
        if options['smooth']:
            lvl1pipe_wf.connect([
                (inputspec, smooth_wf, [('fwhm', 'inputnode.fwhm')]),
                (inputspec, get_gmmask, [('subject_id', 'subj_id'),
                                        ('smooth_gm_mask_template', 'template')]),
                (get_gmmask, mod_gmmask, [('out_file', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (mod_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (fit_mask, smooth_wf, [('out_mask', 'inputnode.mask_file')]),
                (fit_mask, sinker, [('out_mask', 'smoothing_mask')]),
                (get_bold, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [(('outputnode.smoothed_files', unlist), 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
                ])
        else:
            lvl1pipe_wf.connect([
                (get_bold, maskBold, [('out_file', 'img_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
                ])

    lvl1pipe_wf.connect([
        (specify_model, modelfit, [('session_info', 'inputspec.session_info')]),
        (inputspec, sinker, [('subject_id','container'),
                              ('sinker_subs', 'substitutions')]), # creates folder for each subject.
        (make_outdir, sinker, [('new_out_dir', 'base_directory')]),
        (modelfit, sinker, [('outputspec.parameter_estimates', 'model'),
                            ('outputspec.dof_file','model.@dof'), #.@ puts this in the model folder.
                            ('outputspec.copes','model.@copes'),
                            ('outputspec.varcopes','model.@varcopes'),
                            ('outputspec.zfiles','stats'),
                            ('outputspec.pfiles', 'stats.@pfiles'),
                            ('level1design.ev_files', 'design'),
                            ('level1design.fsf_files', 'design.@fsf'),
                            ('modelgen.con_file', 'design.@confile'),
                            ('modelgen.design_cov', 'design.@covmatriximg'),
                            ('modelgen.design_image', 'design.@designimg'),
                            ('modelestimate.logfile', 'design.@log'),
                            ('modelestimate.sigmasquareds', 'model.@resid_sum'),
                            ('modelestimate.fstats', 'stats.@fstats'),
                            ('modelestimate.thresholdac', 'model.@serial_corr'),
                           ])
        ])
    if options['keep_resid']:
        lvl1pipe_wf.connect([
            (modelfit, sinker, [('modelestimate.residual4d', 'model.@resid')
                               ])
            ])
    return lvl1pipe_wf
