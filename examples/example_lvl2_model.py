from jt_modeling import create_lvl2tfce_wf
fwhm_list = [ # use 'none' for no smoothing.
    'none',
    '1.5',
    '6'
            ]
full_cons = {
    '1_instructions_Instructions': [('1_instructions_Instructions', 'T', ['1_instructions_Instructions'], [1])],
    '2_speech_prep_Speech_prep': [('2_speech_prep_Speech_prep', 'T', ['2_speech_prep_Speech_prep'], [1])],
    '3_no_speech_No_speech': [('3_no_speech_No_speech', 'T', ['3_no_speech_No_speech'], [1])],
}
regressors = {
    '1_instructions_Instructions': {'1_instructions_Instructions': [1] * len(subject_list)},
    '2_speech_prep_Speech_prep': {'2_speech_prep_Speech_prep': [1] * len(subject_list)},
    '3_no_speech_No_speech': {'3_no_speech_No_speech': [1] * len(subject_list)}
}
subject_list = [
    'sub-001','sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008',
    'sub-009', 'sub-010','sub-011', 'sub-012', 'sub-013',
#     'sub-014' # Problem with reconstruction of T1.
    'sub-015','sub-016',
#     'sub-017', # Subject dropped out.
#     'sub-018' # High movement. DROP
    'sub-019',
#     'sub-020' # High movement. DROP
#     'sub-021' # High movement. DROP
    'sub-022','sub-023','sub-024','sub-025','sub-026',
#     'sub-027', # Subject dropped out.
#     'sub-028', # No stress task.
#     'sub-029', # Subject dropped out.
    'sub-030',
#     'sub-031' # High movement. DROP
    'sub-032',
#     'sub-033' # High movement. DROP
    'sub-034',
]

######## Masking ################################################
# mask_file = '/home/neuro/atlases/FSMAP/stress/masks/ALIGN_harvardoxford-subcortical_prob_Brain-Stem.nii.gz'
# lvl2_tfce_wf = create_lvl2tfce_wf(mask_file)

######## Workflow setup ################################################
lvl2_tfce_wf = create_lvl2tfce_wf()
# lvl2_tfce_wf.inputs.inputspec.fwhm = '' # setup in iterables below.
# lvl2_tfce_wf.inputs.inputspec.contrast = '' # setup in iterables below.
lvl2_tfce_wf.inputs.inputspec.input_dir = '/home/neuro/data'
lvl2_tfce_wf.inputs.inputspec.output_dir = '/home/neuro/output'
lvl2_tfce_wf.inputs.inputspec.subject_list = subject_list
lvl2_tfce_wf.inputs.inputspec.full_cons = full_cons
lvl2_tfce_wf.inputs.inputspec.con_regressors = regressors
lvl2_tfce_wf.inputs.inputspec.sinker_subs = [
    ('tstat', 'raw_tstat'),
    ('tfce_corrp_raw_tstat', 'tfce_corrected_p')
]

######## Setup surrounding workflow for iteration. ################################################
import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface

def con_dic_to_list(full_cons): # get contrast names from full_cons
   con_list = []
   for entry in list(full_cons.values())[:]:
       con_list.append(entry[0][0])
   return con_list

infosource = pe.Node(IdentityInterface(fields=['fwhm', 'contrast']),
           name='infosource')
infosource.iterables = [('fwhm', fwhm_list),
   ('contrast', con_dic_to_list(full_cons))]

full_wf = pe.Workflow(name='full_wf')
full_wf.connect([
    (infosource, lvl2_tfce_wf, [('fwhm', 'inputspec.fwhm'),
                               ('contrast', 'inputspec.contrast')])])
full_wf.base_dir = '/home/neuro/workdir'

#### Visualize ###########################################################################################
# full_wf.write_graph('simple.dot')
# from IPython.display import Image
# import os
# Image(filename= os.path.join(full_wf.base_dir, 'full_wf','simple.png'))

#### Run model ###########################################################################################
# lvl2_tfce_wf.inputs.randomise.num_perm = 10 # For testing pipeline. Default = 5000
full_wf.run(plugin='MultiProc')
