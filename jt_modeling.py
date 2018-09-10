def create_lvl2tfce_wf():

    '''
    Input [Mandatory]:
    '''

    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function

    ##################  Setup workflow.
    lvl2tfce_wf = pe.Workflow(name='make_ref_img')

    infosource = pe.Node(IdentityInterface(fields=['fwhm', 'contrast']),
                 name='infosource')
                 infosource.iterables = [('contrast', contrast_list),
                       ('fwhm', fwhm_list)]

    ################## Make template
    def get_template(fwhm, contrast, output_dir):
        import os
        # makes template to grab copes files, based on the smoothing kernel being processed.
        if fwhm == 'none':
            con_file = 'cope'+contrast+'.nii.gz'
            template={
                'cope': os.path.join('/home/neuro/data/', 'nosmooth', 'sub-*/model/sub-*',
                         'data', con_file)
            }
            out_path = os.path.join(output_dir, 'nosmooth')
        else:
            fwhm_path = 'fwhm_'+fwhm
            con_file = 'cope'+contrast+'.nii.gz'
            template={
                'cope': os.path.join('/home/neuro/data/', 'smooth', 'sub-*/model/sub-*',
                         fwhm_path, 'data', con_file)
            }
            out_path = os.path.join(output_dir, fwhm)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        return template, out_path

    from nipype.interfaces.utility.wrappers import Function
    make_template = pe.Node(Function(input_names=['fwhm', 'contrast', 'output_dir'],
                                 output_names=['template', 'out_path'],
                                  function=get_template),
                         name='make_template')
    make_template.inputs.output_dir = output_dir
    # make_template.inputs.fwhm # From infosource.

    ################## Get contrast
    def get_con(contrast, full_cons, full_regs):
        con_info = full_cons[contrast]
        reg_info = full_regs[contrast]
        return con_info, reg_info

    get_model_info = pe.Node(Function(input_names=['contrast', 'full_cons', 'full_regs'],
                                   output_names=['con_info', 'reg_info'],
                                   function=get_con),
                          name='get_model_info')
    get_model_info.inputs.full_cons = full_cons
    get_model_info.inputs.full_regs = full_regs

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
    get_copes.inputs.subject_list = subject_list
    # get_copes.inputs.template = template # From make_template.


    ################## Merge into 4d files.
    import nipype.interfaces.fsl as fsl # fsl
    merge_copes = pe.Node(interface=fsl.Merge(dimension='t'), # Changed this from mapnode. TODO make sure this is ok.
                      name='merge_copes')
    # merge_copes.inputs.in_files = copes

    ################## Level 2 design.
    level2model = pe.Node(interface=fsl.MultipleRegressDesign(),
                         name='level2model')
    # level2model.inputs.contrasts # from get_con_info
    # level2model.inputs.regressors # from get_con_info
    wf.connect([(get_model_info, level2model, [('con_info', 'contrasts')]),
                (get_model_info, level2model, [('reg_info', 'regressors')]),
                (level2model, sinker, [('design_con', 'out.@con')]),
    #             (level2model, sinker, [('design_fts', 'out.@fts')]), #TODO - add if F contrast.
                (level2model, sinker, [('design_grp', 'out.@grp')]),
                (level2model, sinker, [('design_mat', 'out.@mat')]),
                ])

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

    ################## Setup datasink.
    from nipype.interfaces.io import DataSink
    import os
    # sinker = pe.Node(DataSink(parameterization=False), name='sinker')
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')
    sinker.inputs.substitutions = sinker_substitutions
    # sinker.inputs.base_directory =


    wf.connect([
        (infosource, make_template, [('fwhm', 'fwhm')]),
        (infosource, make_template, [('contrast', 'contrast')]),
        ])
    wf.connect([
        (infosource, get_model_info, [('contrast', 'contrast')])
        ])
    wf.connect([
        (make_template, get_copes, [('template', 'template')])
        ])
    wf.connect([(merge_copes, sinker, [('merged_file', 'out')]),
               (get_copes, merge_copes, [('out_list', 'in_files')])])
   wf.connect([
       (make_template, sinker, [('out_path', 'base_directory')])
       ])
       
    wf.connect([(merge_copes, randomise, [('merged_file', 'in_file')]), # Changed this from (merge_copes, randomise, [(('merged_file', index_list), 'in_file')]),
                (level2model, randomise, [('design_mat', 'design_mat')]),
                (level2model, randomise, [('design_con', 'tcon')]),
    #             (randomise, sinker, [('f_corrected_p_files', 'out.@f_cor_p')]),
    #             (randomise, sinker, [('f_p_files', 'out.@f_p')]),
    #             (randomise, sinker, [('fstat_files', 'out.@fstat')]),
    #             (randomise, sinker, [('t_p_files', 'out.@t_p')]),
                (randomise, sinker, [('t_corrected_p_files', 'out.@t_cor_p')]),
                (randomise, sinker, [('tstat_files', 'out.@t_stat')]),
               ])
    if mask_file:
        wf.connect([(FLIRT, randomise, [('out_file', 'mask')])]) # TODO - switch input to from workflow.
