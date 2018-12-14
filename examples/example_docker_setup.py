# LVL 1 modeling
docker run -it --rm \
  -v ~/projects/NCI_U01/FSMAP_data/BIDS_fmriprep:/home/neuro/data \
  -v ~/projects/NCI_U01/FSMAP_data/BIDS_modeled:/home/neuro/output \
  -v ~/projects/NCI_U01/scripts:/home/neuro/scripts \
  -v ~/projects/NCI_U01/FSMAP_data/workdir:/home/neuro/workdir \
  -v ~/atlases/mindboggle_atlas/jointfusion_volume_atlases:/home/neuro/atlases \
  -p 8888:8888 \
  jtheriaultpsych/jtnipyutil

# LVL 2 modeling
  docker run -it --rm \
    -v ~/projects/NCI_U01/FSMAP_data/BIDS_modeled:/home/neuro/data \
    -v ~/projects/NCI_U01/FSMAP_data/BIDS_modeled_2lvl:/home/neuro/output \
    -v ~/projects/NCI_U01/scripts:/home/neuro/scripts \
    -v ~/projects/NCI_U01/FSMAP_data/workdir:/home/neuro/workdir \
    -v ~/atlases:/home/neuro/atlases \
    -p 8888:8888 \
    jtheriaultpsych/jtnipyutil

# Once inside the docker image, a jupyter notebook can be opened:
jupyter notebook --port=9999 --no-browser --ip=0.0.0.0 --allow-root
