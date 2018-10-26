docker run --rm kaczmarj/neurodocker:master generate docker \
--base debian:stretch --pkg-manager apt \
--install gcc g++ graphviz tree \
          git vim emacs-nox nano less ncdu \
          tig  \
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
