# Setup outside.
# 'study_dir',
# 'work_dir',
# 'output_dir',
# 'crash_dump',
def create_lvl2tfce_wf(fwhm_list, full_cons):
    '''
    Input [Mandatory]:
        ~~~~~~~~~~ Set as part of function call:
        fwhm_list: list of strings representing smoothing kernels. ITERABLE.
            'None' represents no smoothing.
            e.g. ['none', '1.5', '6']
        full_cons: dictionary of each contrast. ITERABLE.
            Names should match con_regressors.
            Entries in format [('name', 'stat', [condition_list], [weight])]
            e.g. full_cons = {
                '1_instructions_Instructions': [('1_instructions_Instructions', 'T', ['1_instructions_Instructions'], [1])]
                }

        ~~~~~~~~~~~ Set through inputs.inputspec
        input_dir: string, representing directory to level1 data, modeled using TODO.
            e.g. inputs.inputspec.input_dir = '/home/neuro/data/'
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
            mask_file: path to mask file. Must be in same space as functional data.
                see jt_util.create_align_mask_wf
                e.g. inputs.inputspec.mask_file = '/home/neuro/atlases/FSMAP/stress/realigned_masks.amygdala_bl_flirt.nii.gz'
            sinker_subs: list of tuples, each containing a pair of strings.
                These will be sinker substitutions. They will change filenames in the output folder.
                Usually best to run the pipeline once, before deciding on these.
                e.g. inputs.inputspec.sinker_substitutions = [('tstat', 'raw_tstat'),
                       ('tfce_corrp_raw_tstat', 'tfce_corrected_p')]
    '''
    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function

    ##################  Setup workflow.
    lvl2tfce_wf = pe.Workflow(name='make_ref_img')

    inputspec = pe.Node(IdentityInterface(
        fields=['input_dir',
                'output_dir',
                'mask_file',
                'subject_list',
                'con_regressors',
                'sinker_subs',
                ],
        mandatory_inputs=False),
                 name='inputspec')
    inputs.inputspec.fwhm_list = fwhm_list
    inputs.inputspec.full_cons = full_cons

     # Create list from contrast dictionary.
    def con_dic_to_list(full_cons):
        con_list = []
        for entry in list(full_cons.values())[:]:
            con_list.append(entry[0][0])
        return con_list

    infosource = pe.Node(IdentityInterface(fields=['fwhm', 'contrast']),
                name='infosource')
    infosource.iterables = [('fwhm', fwhm_list),
        ('contrast', con_dic_to_list(full_cons))]

    ################## Make template
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
            os.mkdir(out_path)
        return template, out_path

    from nipype.interfaces.utility.wrappers import Function
    make_template = pe.Node(Function(input_names=['fwhm', 'contrast', 'input_dir, ''output_dir'],
                                output_names=['template', 'out_path'],
                                function=get_template),
                        name='make_template')
    # make_template.inputs.output_dir = from inputspec
    # make_template.inputs.input_dir = from inputspec
    # make_template.inputs.fwhm # From infosource.

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
        out_list = []
        for x in glob.glob(list(template.values())[0]):
            if any(subj in x for subj in subject_list):
                out_list.append(x)
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

    ################## Setup Pipeline.
    lvl2tfce_wf.connect([
        (infosource, make_template, [('fwhm', 'fwhm'),
                                    ('contrast', 'contrast')]),
        (inputspec, make_template, [('input_dir', 'input_dir'),
                                    ('output_dir', 'output_dir')]),
        (inputspec, get_model_info, [('full_cons', 'full_cons'),
                                    ('con_regressors', 'full_regs')]),
        (infosource, get_model_info, [('contrast', 'contrast')]),
        (inputspec, get_copes, [('subject_list', 'subject_list')]),
        (make_template, get_copes, [('template', 'template')]),
        (get_copes, merge_copes, [('out_list', 'in_files')]),
        (get_model_info, level2model, [('con_info', 'contrasts')]),
        (get_model_info, level2model, [('reg_info', 'regressors')]),
        (merge_copes, randomise, [('merged_file', 'in_file')]),
        (level2model, randomise, [('design_mat', 'design_mat')]),
        (level2model, randomise, [('design_con', 'tcon')]),
        ])
    if inputs.inputspec.mask_file:
        lvl2tfce_wf.connect([
            (inputspec, randomise, [('mask_file', 'mask')]),
            ])

    ################## Setup datasink.
    from nipype.interfaces.io import DataSink
    import os
    # sinker = pe.Node(DataSink(parameterization=False), name='sinker')
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')

    lvl2tfce_wf.connect([
        (infosource, sinker, [('sinker_subs', 'substitutions')]),
        (make_template, sinker, [('out_path', 'base_directory')]),
        (level2model, sinker, [('design_con', 'out.@con')]),
        (level2model, sinker, [('design_grp', 'out.@grp')]),
        (level2model, sinker, [('design_mat', 'out.@mat')]),
        (randomise, sinker, [('t_corrected_p_files', 'out.@t_cor_p')]),
        (randomise, sinker, [('tstat_files', 'out.@t_stat')]),
        ])
    return lvl2tfce_wf
