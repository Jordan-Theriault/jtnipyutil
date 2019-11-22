def create_corthick_wf():
    import nipype.pipeline.engine as pe # pypeline engine
    import os
    from nipype import IdentityInterface
    from nipype.interfaces.ants.segmentation import KellyKapowski
    from nipype.interfaces.io import DataSink

    corthick_wf = pe.Workflow(name='corthick_wf')

    inputspec = pe.Node(IdentityInterface(
        fields=['seg_file','wmprob_file','out_dir'],
        mandatory_inputs=False), name='inputspec')
    DiReCT = pe.Node(KellyKapowski(), name = 'DiReCt')
    sinker = pe.Node(DataSink(parameterization=True), name='sinker')

    corthick_wf.connect([
        (inputspec, DiReCT, [('seg_file', 'segmentation_image'),
                             ('wmprob_file', 'white_matter_prob_image')]),
        (inputspec, sinker, [('out_dir', 'base_directory')]),
        (DiReCT, sinker, [('cortical_thickness', 'out.@thick'),
                             ('warped_white_matter', 'out.@wm')]),
    ])
    return corthick_wf

import os
# from jtnipyutil.anat import create_corthick_wf

subj = os.environ['SUBJ']
space = os.environ['SPACE']
seg_file = os.path.join('/scratch',
                        'input',
                        subj+'_T1w_space-'+space+'_dtissue.nii.gz')
wmprob_file = os.path.join('/scratch',
                           'input',
                           subj+'_T1w_space-'+space+'_class-WM_probtissue.nii.gz')
out_dir = os.path.join('/scratch',
                       'output')

full_wf = create_corthick_wf()
full_wf.inputs.inputspec.seg_file = seg_file
full_wf.inputs.inputspec.wmprob_file = wmprob_file
full_wf.inputs.inputspec.out_dir = out_dir
full_wf.base_dir = '/scratch/work/'
full_wf.run(plugin='MultiProc')
