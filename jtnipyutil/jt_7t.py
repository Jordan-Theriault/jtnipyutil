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





# To create template for PAG, use volume-based method (FNIRT in fsl)
# Normalization step is probably normalizing from the MNI template brain to their individual brain. Ask Phil though.
