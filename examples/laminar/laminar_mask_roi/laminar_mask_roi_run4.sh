#!/bin/tcsh
setenv SUBJ sub-$1
setenv TASK rest_run-04
setenv PROJNAME nighres_roi
setenv DEPTH ${SUBJ}_space-MNI152NLin2009cAsym_depth_layering-depth.nii.gz
setenv IMG ${SUBJ}_task-${TASK}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
setenv CONFOUND ${SUBJ}_task-${TASK}_bold_confounds.tsv
setenv SINGULARITY /usr/bin/singularity
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/nighres/nighres-2019-05-07-9ae9dfd9c326.simg

setenv LAMINAR_DATA /autofs/cluster/iaslab/users/jtheriault/FSMAP/laminar
setenv RAW_DATA /autofs/cluster/iaslab/FSMAP/FSMAP_data/BIDS_fmriprep/fmriprep/ses-01
setenv SCRIPTS /autofs/cluster/iaslab/users/jtheriault/FSMAP/laminar/scripts/laminar_mask_roi
setenv OUTPUT /autofs/cluster/iaslab/users/jtheriault/FSMAP/laminar/roi_data

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/data/rois
cp -ra $RAW_DATA/$SUBJ/func/$IMG /scratch/$USER/$SUBJ/$PROJNAME/data/
cp -ra $RAW_DATA/$SUBJ/func/$CONFOUND /scratch/$USER/$SUBJ/$PROJNAME/data/
cp -ra $LAMINAR_DATA/depth/$DEPTH /scratch/$USER/$SUBJ/$PROJNAME/data/
cp -ra $LAMINAR_DATA/rois/*.nii /scratch/$USER/$SUBJ/$PROJNAME/data/rois/

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/scripts
cp -ra $SCRIPTS/* /scratch/$USER/$SUBJ/$PROJNAME/scripts/
chmod +x /scratch/$USER/$SUBJ/$PROJNAME/scripts/*

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/nighres

cd /scratch/$USER

$SINGULARITY exec \
--bind "/scratch:/scratch" \
$IMAGE \
/scratch/$USER/$SUBJ/$PROJNAME/scripts/laminar_startup_mask_roi.sh

rsync -ra /scratch/$USER/$SUBJ/$PROJNAME/nighres/depth10* $OUTPUT/
rsync -ra /scratch/$USER/$SUBJ/$PROJNAME/nighres/depth3* $OUTPUT/

rm -r /scratch/$USER/$SUBJ/$PROJNAME
exit
