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
                'cope': os.path.join(input_dir, 'nosmooth', 'sub-*/model/sub-*',
                         'data', con_file)
            }
            out_path = os.path.join(output_dir, 'nosmooth')
        else:
            fwhm_path = 'fwhm_'+fwhm
            con_file = 'cope'+contrast+'.nii.gz'
            template={
                'cope': os.path.join(input_dir, 'smooth', 'sub-*/model/sub-*',
                        fwhm_path, 'data', con_file)
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
        def fit_msk(mask_file, ref_file):
            import numpy as np
            import nibabel as nib
            import os
            from scipy.ndimage import zoom
            mask = nib.load(mask_file)
            ref = nib.load(ref_file)
            if mask.shape != ref.shape[0:3]:
                interp_dims = np.array(ref.shape[0:3])/np.array(mask.shape)
                data = zoom(mask.get_data(), interp_dims.tolist()) # interpolate mask to native space.
                out_mask = nib.Nifti1Image(data, ref.affine, ref.header)
                out_mask.header['cal_max'] = np.max(data) # adjust min and max header info.
                out_mask.header['cal_min'] = np.min(data)
                nib.save(out_mask, os.path.abspath(mask_file + '_fit.nii.gz'))
                out_mask = os.path.abspath(mask_file + '_fit.nii.gz')
            return out_mask
    if mask:
        fit_mask = pe.Node(Function(
            input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'],
            function=fit_msk),
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

        options:

        ~~~~~~~~~~~ Set through inputs.inputspec

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
        for x in glob.glob(list(template.values())[0]):
            if subj_id in x:
                temp_list.append(x)
        for file in temp_list: # ensure no duplicate entries.
            if file not in out_list:
                out_list.append(file)
        assert (len(out_list) == 1), 'Each combination of template and subject ID should return 1 file. Either no data, or multiple files were found.'
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
        def fit_msk(mask_file, ref_file):
            import numpy as np
            import nibabel as nib
            import os
            from scipy.ndimage import zoom
            mask = nib.load(mask_file)
            ref = nib.load(ref_file)
            if mask.shape != ref.shape[0:3]:
                interp_dims = np.array(ref.shape[0:3])/np.array(mask.shape)
                data = zoom(mask.get_data(), interp_dims.tolist()) # interpolate mask to native space.
                out_mask = nib.Nifti1Image(data, ref.affine, ref.header)
                out_mask.header['cal_max'] = np.max(data) # adjust min and max header info.
                out_mask.header['cal_min'] = np.min(data)
                nib.save(out_mask, os.path.abspath(mask_file + '_fit.nii.gz'))
                out_mask = os.path.abspath(mask_file + '_fit.nii.gz')
            return out_mask
        fit_mask = pe.Node(Function(
            input_names=['mask_file', 'ref_file'],
            output_names=['out_mask'],
            function=fit_msk),
                            name='fit_mask')

    ################## Setup confounds
    def get_terms(confound_file, noise_transforms, noise_regressors, TR, options):
        # # Add time derivs, quadratic terms, and quad time derives if requested.
        import numpy as np
        import pandas as pd
        df_cf = pd.DataFrame(pd.read_csv(confound_file, sep='\t', parse_dates=False))
        confounds = df_cf[noise_regressors] # for output
        transfrm_list = []
        for entry in noise_regressors: # get entries marked with *, indicating they should be transformed.
            if '*' in entry:
                transfrm_list.append(entry.replace('*', ''))
        transfrmd_cnfds = df_cf[transfrm_list] # for transforms
        TR_time = pd.Series(np.arange(0.0, TR*transfrmd_cnfds.shape[0], TR)) # time series for derivatives.
        if 'quad' in noise_transforms:
            quad = np.square(df_cf[transfrmd_cnfds])
            confounds = confounds.join(quad, rsuffix='_quad')
        if 'tderiv' in noise_transforms:
            tderiv = pd.DataFrame(pd.Series(np.gradient(transfrmd_cnfds[col]), TR_time)
                                  for col in transfrmd_cnfds).T
            confounds = confounds.join(tderiv, rsuffix='_tderiv')
        if 'quadtderiv' in noise_transforms:
            quadtderiv = np.square(tderiv)
            confounds = confounds.join(quadtderiv, rsuffix='_quadtderiv')
        if options['remove_steadystateoutlier']:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^NonSteadyStateOutlier')]])
        if options['ICA_AROMA']:
            confounds = confounds.join(df_cf[df_cf.columns[df_cf.columns.to_series().str.contains('^AROMAAggrComp')]])
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
    from nipype.interfaces.fsl.maths import ApplyMask
    maskBold = pe.Node(ApplyMask(),
                      name='maskBold')
    # maskBold.inputs.in_file # From get_bold
    # maskBold.inputs.mask_file # From get_mask

    ################## Despike
    from nipype.interfaces.afni import Despike
    despike = pe.Node(Despike(),
                      name='despike')
    # despike.inputs.in_file = # From Mask

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
        out = not input
        return out

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
                (get_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (fit_mask, mod_gmmask, [('out_mask', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (mod_gmmask, smooth_wf, [('out_file', 'inputnode.mask_file')]),
                (mod_gmmask, sinker, [('out_file', 'smoothing_mask')]),
                (despike, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [('outputnode.smoothed_files', 'in_file')]),
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
                (get_gmmask, fit_mask, [('out_file', 'mask_file')]),
                (fit_mask, mod_gmmask, [('out_mask', 'in_file')]),
                (inputspec, mod_gmmask, [('gmmask_args', 'args')]),
                (get_bold, fit_mask, [('out_file', 'ref_file')]),
                (mod_gmmask, smooth_wf, [('out_file', 'inputnode.mask_file')]),
                (mod_gmmask, sinker, [('out_file', 'smoothing_mask')]),
                (get_bold, smooth_wf, [('out_file', 'inputnode.in_files')]),
                (smooth_wf, maskBold, [('outputnode.smoothed_files', 'in_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
                ])
        else:
            lvl1pipe_wf.connect([
                (get_bold, maskBold, [('out_file', 'in_file')]),
                (maskBold, specify_model, [('out_file', 'functional_runs')]),
                (maskBold, modelfit, [('out_file', 'inputspec.functional_data')])
                ])

    lvl1pipe_wf.connect([
        (specify_model, modelfit, [('session_info', 'inputspec.session_info')]),
        (inputspec, sinker, [('subject_id','container'),
                              ('sinker_subs', 'substitutions')]), # creates folder for each subject.
        (make_outdir, sinker, [('new_out_dir', 'base_directory')]),
        (modelfit, sinker, [('outputspec.dof_file','model.@dof'), #.@ puts this in the par folder.
                            ('outputspec.parameter_estimates', 'model'),
                            ('outputspec.copes','model.@copes'),
                            ('outputspec.varcopes','model.@varcopes'),
                            ('level1design.ev_files', 'design'),
                            ('level1design.fsf_files', 'design.@fsf'),
                            ('modelgen.con_file', 'design.@confile'),
                            ('modelgen.design_cov', 'design.@covmatriximg'),
                            ('modelgen.design_image', 'design.@designimg'),
                            ('modelestimate.logfile', 'design.@log'),
                            ('modelestimate.residual4d', 'model.@resid'),
                            ('modelestimate.fstats', 'stats.@fstats'),
                           ])
        ])

    if options['run_contrasts']:
        lvl1pipe_wf.connect([
            (modelfit, sinker, [('outputspec.zfiles','stats'),
                                ('outputspec.pfiles', 'stats.@pfiles')])])
    return lvl1pipe_wf
