#!/bin/tcsh
setenv DATA /autofs/cluster/iaslab/FSMAP/FSMAP_data
setenv SCRIPTS /autofs/cluster/iaslab/users/jtheriault/FSMAP/laminar/scripts
setenv OUTPUT /autofs/cluster/iaslab/users/jtheriault/FSMAP/laminar/depth
setenv SUBJ sub-$1
setenv PROJNAME nighres_cort_depth
setenv TASK rest
setenv SPACE MNI152NLin2009cAsym
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/nighres/nighres-2019-05-07-9ae9dfd9c326.simg
setenv SINGULARITY /usr/bin/singularity

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/BIDS_fmriprep/$SUBJ/func
rsync -ra $DATA/BIDS_fmriprep/fmriprep/ses-01/$SUBJ/func/*task-$TASK*space-$SPACE*.nii.gz \
/scratch/$USER/$SUBJ/$PROJNAME/BIDS_fmriprep/$SUBJ/func/
rsync -ra $DATA/BIDS_fmriprep/fmriprep/ses-01/$SUBJ/anat \
/scratch/$USER/$SUBJ/$PROJNAME/BIDS_fmriprep/$SUBJ/

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/work/scripts
cp -ra $SCRIPTS/* /scratch/$USER/$SUBJ/$PROJNAME/work/scripts/
chmod +x /scratch/$USER/$SUBJ/$PROJNAME/work/scripts/*

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/nighres
cd /scratch/$USER

$SINGULARITY exec \
--bind "/scratch:/scratch" \
$IMAGE \
/scratch/$USER/$SUBJ/$PROJNAME/work/scripts/startup_cortical_depth.sh

rsync -ra /scratch/$USER/$SUBJ/$PROJNAME/nighres/$SUBJ*depth* $OUTPUT/
rsync -ra /scratch/$USER/$SUBJ/$PROJNAME/nighres/$SUBJ*dWM* $OUTPUT/

rm -r /scratch/$USER/$SUBJ/$PROJNAME
exit
