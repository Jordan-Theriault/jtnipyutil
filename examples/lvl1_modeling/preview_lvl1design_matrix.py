import os, sys
import nipype.pipeline.engine as pe # pypeline engine
import jynipyutil.model.create_lvl1design_wf
from nipype import IdentityInterface

sinker_subs = [('subject_id_',''), # AROMA Setup
                        ('_fwhm','fwhm'),
                        ('_sub','sub'),
                        (' ','_'),
                       ]
########## CREATE MODEL AND SET INPUTSPEC ########################################
options = {'remove_steadystateoutlier': True,
           'ICA_AROMA': False,
           'poly_trend': 1, # no intercept.
           'dct_basis': None}
# options = {'remove_steadystateoutlier': True,
#            'ICA_AROMA': False,
#            'poly_trend': None, # no intercept.
#            'dct_basis': 405}

model_wf = create_lvl1design_wf(options)
model_wf.inputs.inputspec.output_dir = "/home/neuro/workdir/2020-04-27_level1design/output"
model_wf.inputs.inputspec.design_col = 'trial_type'
model_wf.inputs.inputspec.noise_regressors = ['CSF', 'WhiteMatter', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
model_wf.inputs.inputspec.noise_transforms = []
model_wf.inputs.inputspec.TR = 2.34  # In seconds, ensure this is a float
model_wf.inputs.inputspec.hpf_cutoff = 0.
# model_wf.inputs.inputspec.conditions = ['Instructions', 'Speech_prep', 'No_speech', 'Pre_Baseline'] # parameter to model from task file.
model_wf.inputs.inputspec.conditions = ['Instructions', 'Speech_prep', 'No_speech'] # parameter to model from task file.
model_wf.inputs.inputspec.bases = {'dgamma':{'derivs': False}} # Base function for hemodynamic model. dgamma  = double gamma.
#For more options, see Level1Design at https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.fsl/model.html
model_wf.inputs.inputspec.sinker_subs = sinker_subs
model_wf.inputs.inputspec.sample_bold = ['/home/neuro/workdir/2020-04-27_level1design/data/sub-001_task-stress_bold_space-MNI152NLin2009cAsym_preproc.nii.gz']
model_wf.inputs.inputspec.task_template = {'task': '/home/neuro/workdir/2020-04-27_level1design/events/sub-*_task-stress_events.tsv'}
model_wf.inputs.inputspec.confound_template = {'confound': '/home/neuro/workdir/2020-04-27_level1design/confounds/sub-*_task-stress_bold_confounds.tsv'}
model_wf.inputs.inputspec.proj_name = 'stress_DCT'
# subject_list = ['sub-002']
subject_list = ['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-009',
               'sub-010', 'sub-011', 'sub-012', 'sub-015', 'sub-016', 'sub-019', 'sub-022',
               'sub-023', 'sub-024', 'sub-026', 'sub-030', 'sub-034', 'sub-037'] # each subject ID, combined with the template, should specify ONE file.

########## SETUP ITERABLES ########################################

infosource = pe.Node(IdentityInterface(fields=['fwhm', 'subject_id']),
           name='infosource')
infosource.iterables = [('subject_id', subject_list)]
# infosource.iterables = [('subject_id', subject_list)] # If no smoothing.

full_model_wf = pe.Workflow(name='full_model_wf')
full_model_wf.connect([
    (infosource, model_wf, [('subject_id', 'inputspec.subject_id')])])
# full_model_wf.connect([
#     (infosource, model_wf, [('subject_id', 'inputspec.subject_id')])]) # If no smoothing.
full_model_wf.base_dir = '/home/neuro/workdir/2020-04-27_level1design/'
full_model_wf.crash_dump = '/home/neuro/workdir/2020-04-27_level1design/crashdump'

########## RUN ##################################################
full_model_wf.run(plugin='MultiProc')
