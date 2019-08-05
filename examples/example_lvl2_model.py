from jtnipyutil.model import create_lvl2tfce_wf



fwhm_list = [
    'none',
    '1.5',
    '6',
            ]
full_cons = {
    '1_instructions': [('1_instructions', 'T', ['1_instructions'], [1])],
    '2_speech_prep': [('2_speech_prep', 'T', ['2_speech_prep'], [1])],
    '3_no_speech': [('3_no_speech', 'T', ['3_no_speech'], [1])],
}

subject_list = ['sub-001','sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010','sub-011','sub-012','sub-013',
    'sub-015','sub-016','sub-019','sub-022','sub-023','sub-024','sub-025','sub-026','sub-030','sub-032','sub-034','sub-037', 'sub-039',
    'sub-043', 'sub-044', 'sub-045', 'sub-048', 'sub-049', 'sub-050', 'sub-052', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-057',
    'sub-058', 'sub-059', 'sub-060', 'sub-062',  'sub-064', 'sub-065', 'sub-066', 'sub-067', 'sub-068',
    'sub-070', 'sub-071', 'sub-072', 'sub-073', 'sub-074', 'sub-076', 'sub-078', 'sub-081', 'sub-082', 'sub-083', 'sub-084', 'sub-085', 'sub-086', 'sub-087',
#     'sub-014' # Unable to preprocess
#     'sub-017', # Subject dropped out.
#     'sub-018' # High movement. DROP
#     'sub-020' # High movement. DROP
#     'sub-021' # High movement. DROP
#     'sub-027', # Subject dropped out.
#     'sub-028', # No stress task.
#     'sub-029', # Subject dropped out.
#     'sub-031' # High movement. DROP
#     'sub-033' # High movement. DROP
#     'sub-069' # High Movement. DROP
]
# TODO
# model 'sub-061', 'sub-063','sub-080',

lvl2_tfce_wf = create_lvl2tfce_wf()
# lvl2_tfce_wf.inputs.inputspec.fwhm = ''
# lvl2_tfce_wf.inputs.inputspec.contrast = ''
lvl2_tfce_wf.inputs.inputspec.input_dir = '/scratch/data'
lvl2_tfce_wf.inputs.inputspec.output_dir = '/scratch/output'
lvl2_tfce_wf.inputs.inputspec.subject_list = subject_list
lvl2_tfce_wf.inputs.inputspec.full_cons = full_cons
lvl2_tfce_wf.inputs.inputspec.con_regressors = {
    '1_instructions': {'1_instructions': [1] * len(subject_list)},
    '2_speech_prep': {'2_speech_prep': [1] * len(subject_list)},
    '3_no_speech': {'3_no_speech': [1] * len(subject_list)}
}
lvl2_tfce_wf.inputs.inputspec.sinker_subs = [
    ('tstat', 'raw_tstat'),
    ('tfce_corrp_raw_tstat', 'tfce_corrected_p')
]

####### MASK #####################
# lvl2_tfce_wf.inputs.inputspec.mask_file = '/home/neuro/atlases/FSMAP/stress/masks/amygdala_bl.nii.gz'

def con_dic_to_list(full_cons):
   con_list = []
   for entry in list(full_cons.values())[:]:
       con_list.append(entry[0][0])
   return con_list

import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface

infosource = pe.Node(IdentityInterface(fields=['fwhm', 'contrast']),
           name='infosource')
infosource.iterables = [('fwhm', fwhm_list),
   ('contrast', con_dic_to_list(full_cons))]

full_wf = lvl2tfce_wf = pe.Workflow(name='full_wf')
full_wf.connect([
    (infosource, lvl2_tfce_wf, [('fwhm', 'inputspec.fwhm'),
                               ('contrast', 'inputspec.contrast')])])
full_wf.base_dir = '/scratch/wrkdir'

#### Visualize ################
# full_wf.write_graph('simple.dot')
# from IPython.display import Image
# import os
# Image(filename= os.path.join(full_wf.base_dir, 'full_wf','simple.png'))
full_wf.run(plugin='MultiProc')
