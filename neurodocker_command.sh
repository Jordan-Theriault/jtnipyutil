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
--miniconda create_env=py35 \
  conda_install="python=3.5 jupyter jupyterlab jupyter_contrib_nbextensions
                 traits pandas matplotlib scikit-learn seaborn mkl nose sphinx
                 theano pygpu" \
  pip_install="https://github.com/nipy/nipype/tarball/master
              https://github.com/INCF/pybids/tarball/master
              nltools nilearn datalad[full] nipy duecredit niwidgets
              mne deepdish hypertools ipywidgets pynv six nibabel joblib
              parameterized
              git+https://github.com/poldracklab/fmriprep.git
              git+https://github.com/poldracklab/niworkflows.git
              git+https://github.com/pymc-devs/pymc3
              git+https://github.com/PsychoinformaticsLab/nipymc" \
  activate=True \
--copy jtnipyutil /opt/miniconda-latest/envs/py35/lib/python3.5/site-packages/jtnipyutil
