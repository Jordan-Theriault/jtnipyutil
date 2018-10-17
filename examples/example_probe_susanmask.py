import nipype.pipeline.engine as pe # pypeline engine
import nipype.interfaces.fsl as fsl
mod_gmmask = pe.Node(fsl.maths.MathsCommand(),
                        name='mod_gmmask')
mod_gmmask.inputs.in_file = '/home/neuro/data/sub-001/anat/sub-001_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz'
mod_gmmask.inputs.args = '-thr .5 -bin -kernel sphere 1.5 -dilM' # parameters here can be varied until we find a mask that is appropriate.
mod_gmmask.base_dir = '/home/neuro/workdir'
mod_gmmask.run()
