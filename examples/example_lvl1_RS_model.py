import os, sys
from jtnipyutil.model import create_lvl1pipe_wf
# from jtnipyutil.util import combine_runs
import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface
import numpy as np
import pandas as pd
import os
import nibabel as nib


########## SINKER INFO ########################################
# These will clean up the output. You may need to run the pipeline once to know what to put here.
# But if you run-reun the pipeline, old steps are pulled from the cache, so it should rerun quickly.

# sinker_subs = [('subject_id_',''), # AROMA Setup
#                         ('_modelestimate0','data'),
#                         ('_fwhm','fwhm'),
#                         ('_sub','sub'),
#                         (' ','_'),
#                         ('cope1', 'cope1_'+params[0]),
#                         ('cope2', 'cope2_'+params[1]),
#                         ('cope3', 'cope3_'+params[2]),
#                         ('zstat1', 'zstat1_'+params[0]),
#                         ('zstat2', 'zstat2_'+params[1]),
#                         ('zstat3', 'zstat3_'+params[2]),
#                         ('bold_space-MNI152NLin2009cAsym_preproc_masked_maths', 'SUSAN_mask'),
#                         ('pe1', 'pe1_instructions'),
#                         ('pe2', 'pe2_speech_prep'),
#                         ('pe3', 'pe3_no_speech'),
#                         ('res4d', '4dresiduals'),
#                        ]
sinker_subs = []
########## CREATE MODEL AND SET INPUTSPEC ########################################
options = {'remove_steadystateoutlier': True,
           'smooth': True,
           'censoring': 'despike',
           'ICA_AROMA': False,
          'run_contrasts': False,
          'keep_resid': True}

model_wf = create_lvl1pipe_wf(options)
model_wf.inputs.inputspec.input_dir = "/scratch/data/"
model_wf.inputs.inputspec.output_dir = '/scratch/output'
model_wf.inputs.inputspec.design_col = 'trial_type'
model_wf.inputs.inputspec.noise_regressors = ['csf', 'white_matter', 'trans_x*', 'trans_y*', 'trans_z*', 'rot_x*', 'rot_y*', 'rot_z*']
model_wf.inputs.inputspec.noise_transforms = ['quad', 'tderiv', 'quadtderiv']
model_wf.inputs.inputspec.TR = 2.  # In seconds, ensure this is a float
model_wf.inputs.inputspec.FILM_threshold = 1 # Threshold for FILMGLS.  1000: p<=.001, 1: p <=1, i.e. unthresholded.
model_wf.inputs.inputspec.hpf_cutoff = 128.
model_wf.inputs.inputspec.params = [] # parameter to model from task file.
model_wf.inputs.inputspec.contrasts = []
model_wf.inputs.inputspec.bases = {'dgamma':{'derivs': False}} # For more options, see Level1Design at https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.fsl/model.html
model_wf.inputs.inputspec.model_serial_correlations = True # Include Pre-whitening, deals with autocorrelation
model_wf.inputs.inputspec.sinker_subs = sinker_subs #TODO - run one and check this.
model_wf.inputs.inputspec.bold_template = {'bold': '/scratch/data/sub-*/func/sub-*_task-trag_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'}
model_wf.inputs.inputspec.mask_template = {'mask': '/scratch/data/sub-*/func/sub-*_task-trag_run-*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'}
model_wf.inputs.inputspec.task_template = {'task': '/scratch/data/sub-*/func/sub-*_task-trag_run-*_events.tsv'}
model_wf.inputs.inputspec.confound_template = {'confound': '/scratch/data/sub-*/func/sub-*_task-trag_run-*_desc-confounds_regressors.tsv'}
model_wf.inputs.inputspec.smooth_gm_mask_template = {'gm_mask': '/scratch/data/sub-*/anat/sub-*_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz'}
model_wf.inputs.inputspec.gmmask_args =  '-thr .5 -bin -kernel gauss 1 -dilM' # FSL Math command to adjust grey matter for susan smoothing mask.
model_wf.inputs.inputspec.proj_name = 'timecourse'

# model_wf.inputspec.subject_id = # Could use 'sub-001', or use iterables below.
# model_wf.inputspec.fwhm = # could use 1.5, or use iterables below.
subject_list = [sys.argv[1]+'_task-trag_run-01',
                sys.argv[1]+'_task-trag_run-02',
                sys.argv[1]+'_task-trag_run-03',
                sys.argv[1]+'_task-trag_run-04',
                sys.argv[1]+'_task-trag_run-05',
                sys.argv[1]+'_task-trag_run-06'] # each subject ID, combined with the template, should specify ONE file.
fwhm_list = [5.] # Smoothing parameters. If options['smooth'] is false, then leave empty and remove fwhm_list from iterables below.

########## ALTERNATIVE REGRESSOR EXAMPLE: ICA AROMA ########################################

# options = {'remove_steadystateoutlier': True,
#            'smooth': True,
#            'censoring': '',
#            'ICA_AROMA': True,
#           'run_contrasts': False,
#           'keep_resid': True}
# noise_regressors = [] # from fmriprep bold_confounds header.
# noise_transforms = []

########## SETUP ITERABLES ########################################

infosource = pe.Node(IdentityInterface(fields=['fwhm', 'subject_id']),
           name='infosource')
infosource.iterables = [('fwhm', fwhm_list),
   ('subject_id', subject_list)]
# infosource.iterables = [('subject_id', subject_list)] # If no smoothing.

full_model_wf = pe.Workflow(name='full_model_wf')
full_model_wf.connect([
    (infosource, model_wf, [('fwhm', 'inputspec.fwhm'),
                            ('subject_id', 'inputspec.subject_id')])])
# full_model_wf.connect([
#     (infosource, model_wf, [('subject_id', 'inputspec.subject_id')])]) # If no smoothing.
full_model_wf.base_dir = '/scratch/wrkdir/timecourse'
full_model_wf.crash_dump = '/scratch/wrkdir/crashdump'

########## Visualize ##################################################
# full_model_wf.write_graph('simple.dot')
# from IPython.display import Image
# import os
# Image(filename= os.path.join(full_model_wf.base_dir, 'full_model_wf','simple.png'))

########## RUN ##################################################
full_model_wf.run(plugin='MultiProc')
