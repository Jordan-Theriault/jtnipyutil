from jtnipyutil.model import create_lvl2tfce_wf

import os
import nipype.pipeline.engine as pe # pypeline engine
from nipype import IdentityInterface

full_cons = {
    '1_instructions': [('1_instructions', 'T', ['1_instructions'], [1])],
    '2_speech_prep': [('2_speech_prep', 'T', ['2_speech_prep'], [1])],
    '3_no_speech': [('3_no_speech', 'T', ['3_no_speech'], [1])],
}

subject_list = ['sub-001','sub-002','sub-003','sub-004','sub-005']

lvl2_tfce_wf = create_lvl2tfce_wf()
lvl2_tfce_wf.inputs.inputspec.proj_name = os.environ['FWHM']

# NOTE: modify this according to the output of level 1 modeling.
if os.environ['FWHM'] == 'nosmooth':
    lvl2_tfce_wf.inputs.inputspec.copes_template = '/scratch/data/nosmooth/sub-*/model/sub-*/_modelestimate0/cope*nii.gz'
elif os.environ['FWHM'] == '1.5':
    lvl2_tfce_wf.inputs.inputspec.copes_template = '/scratch/data/smooth/sub-*/model/fwhm_1.5sub-*/_modelestimate0/cope*nii.gz'
elif os.environ['FWHM'] == '6':
    lvl2_tfce_wf.inputs.inputspec.copes_template = '/scratch/data/smooth/sub-*/model/fwhm_6.0sub-*/_modelestimate0/cope*nii.gz'
else:
    print('cannot match a valid template to the FWHM given. Try again.')
    exit()
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

infosource = pe.Node(IdentityInterface(fields=['contrast']),
           name='infosource')
infosource.iterables = [('contrast', con_dic_to_list(full_cons))]

full_wf = pe.Workflow(name='full_wf')
full_wf.base_dir = '/scratch/wrkdir'

full_wf.connect([
    (infosource, lvl2_tfce_wf, [('contrast', 'inputspec.contrast')])])

#### Visualize ################
# full_wf.write_graph('simple.dot')
# from IPython.display import Image
# import os
# Image(filename= os.path.join(full_wf.base_dir, 'full_wf','simple.png'))
full_wf.run(plugin='MultiProc')
