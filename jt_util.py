def create_ref_img_wf():
    '''
    Creates a workflow, which creates a reference image, averaging across all functional files given.

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

    TODO
    '''

    import nipype.pipeline.engine as pe # pypeline engine
    import nipype.interfaces.fsl as fsl
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.utility.wrappers import Function

    # Setup workflow.
    ref_img_wf = pe.Workflow(name='make_ref_img')
    inputspec = pe.Node(IdentityInterface(
        fields=['subject_list', 'template']),
                 name='inputspec')
    outputspec = pe.Node(IdentityInterface(
        fields=['mean_img']),
                        name='outputspec')

    # get functional files.
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

    # average each functional file, along 4th dimension.
    # def mean_4d(in_file):
    #     import nibabel as nib
    #     from nilearn.image import mean_img
    #     mean_img = mean_img(in_file)
    #     nib.save(mean_img, in_file+'_mean.nii.gz')
    #     out_file = nib.load(in_file+'_mean.nii.gz')
    #     return out_file

    mean_imgs = pe.MapNode(fsl.maths.MeanImage(),
        iterfield = ['in_file'],
        name='mean_imgs')

    # combine all averaged functional images.
    merge_imgs = pe.Node(interface=fsl.Merge(dimension='t'),
                       name='merge_imgs')
#     merge_func.inputs.in_files = # from mean_func
    merge_imgs.inputs.dimension = 't'

    # Then average all average functional images.
    grand_mean = pe.Node(fsl.maths.MeanImage(),
        iterfield = ['in_file'],
        name='grand_mean')


    ref_img_wf.connect([(inputspec, get_imgs, [('subject_list', 'subject_list')]),
                    (inputspec, get_imgs, [('template', 'template')]),
                    (get_imgs, mean_imgs, [('out_list', 'in_file')]),
                    (mean_imgs, merge_imgs, [('out_file', 'in_files')]),
                    (merge_imgs, grand_mean, [('merged_file', 'in_file')]),
                    (grand_mean, outputspec, [('out_file', 'mean_img')]),
                    ])
    return ref_img_wf



# def create_align_mask_wf():
#
#     #TODO work on this
#         ################ Align Mask WF.
#     align_ref = pe.Node(interface=fsl.FLIRT(),
#                        name='align_ref')
#     align_ref.inputs.in_file = mask_space_brain #TODO - from inputspec
#     align_ref.inputs.reference = # from make_mean_ref.outputs.out_file
#
#     align_mask = pe.Node(interface=fsl.FLIRT(),
#                         name='align_mask')
#     align_mask.inputs.in_file = mask_file #TODO - from inputspec
#     align_mask.inputs.reference = # from make_mean_ref.outputs.out_file
#     align_mask.inputs.apply_xfm = True
#     align_mask.inputs.in_matrix_file = # From align_ref.outputs.out_matrix_file
