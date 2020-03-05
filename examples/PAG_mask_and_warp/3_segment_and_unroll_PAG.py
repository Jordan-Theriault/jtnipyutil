'''
docker run -it --rm \
-v ~/projects/NCI_U01/FSMAP_data/workdir2:/home/neuro/workdir \
-p 8888:8888 \
jtheriaultpsych:jtnipyutil

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
'''


from jtnipyutil.fsmap import segment_and_unroll_PAG
import os

root = '/home/neuro/workdir/2019-12-04_PAG_lvl2'
PAG_file = os.path.join(root, 'template/Template_6.nii')
con_file = os.path.join(root, 'data/dartel/nosmooth/_contrast_2_speech_prep/randomise_raw_tstat1.nii.gz')
con_name = 'speech_prep'
out_dir = os.path.join(root, 'output')

segment_and_unroll_PAG(PAG_file, con_file, con_name, out_dir, thresh = .2)
