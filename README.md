# jtnipyutil
fMRI Python utility functions

# Nipype Docker Image

## Usage Example:
docker run -it --rm -p 8888:8888 \
  -v ~/path/to/data/dir:/home/neuro/data \
  -v ~/path/to/output/dir:/home/neuro/output \
  -v ~/path/to/scripts/dir:/home/neuro/scripts \
  -v ~/path/to/working/dir:/home/neuro/workdir \
  jtheriaultpsych/jtnipyutil

## Once open,  jupyter workbook can be opened with the folling:
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

## This will generate a key, like this:
http://97afb4516f4b:8888/?token=50f351a966bb65b147807bddee19b5aa53cff089cf5afe73&token=50f351a966bb65b147807bddee19b5aa556asdfhjadsf741241

## Change the characters at the beginning and paste it into a web browser, e.g.:
http://localhost:8888/?token=50f351a966bb65b147807bddee19b5aa53cff089cf5afe73&token=50f351a966bb65b147807bddee19b5aa556asdfhjadsf741241

## Navigate from the root to whatever folder you set up.



## Neurodocker command:
docker run --rm kaczmarj/neurodocker:master generate docker \
--base debian:stretch --pkg-manager apt \
--install gcc g++ graphviz tree bsdtar \
          git vim emacs-nox nano less ncdu \
          tig  \
--run "export tar='bsdtar'" \
--fsl version=5.0.11 \
--ants version=2.2.0 \
--convert3d version=1.0.0 \
--freesurfer version=6.0.0-min \
--afni version=latest \
--spm version=r7219 \
--miniconda create_env=py36 \
  conda_install="python=3.6 jupyter jupyterlab jupyter_contrib_nbextensions
                 traits pandas matplotlib scikit-learn seaborn" \
  pip_install="https://github.com/nipy/nipype/tarball/master
               https://github.com/INCF/pybids/tarball/master
               nltools nilearn datalad[full] nipy duecredit niwidgets
               mne deepdish hypertools ipywidgets pynv six nibabel joblib
               git+https://github.com/poldracklab/fmriprep.git
               git+https://github.com/poldracklab/niworkflows.git" \
  activate=True \
--copy jtnipyutil /opt/miniconda-latest/envs/py36/lib/python3.6/site-packages/jtnipyutil

