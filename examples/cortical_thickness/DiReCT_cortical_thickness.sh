#!/bin/tcsh
setenv DATA /autofs/cluster/iaslab/FSMAP/FSMAP_data
setenv SCRIPTS /autofs/cluster/iaslab/users/jtheriault/FSMAP/scripts/cortical_thickness
setenv OUTPUT /autofs/cluster/iaslab/users/jtheriault/FSMAP/DiReCT
setenv SUBJ sub-$1
setenv PROJNAME DiReCt_cortical_thickness
setenv SPACE MNI152NLin2009cAsym
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/jtnipyutil/jtnipyutil-2019-01-03-4cecb89cb1d9.simg
setenv SINGULARITY /usr/bin/singularity

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/input
rsync -ra $DATA/BIDS_fmriprep/fmriprep/ses-01/$SUBJ/anat/* \
/scratch/$USER/$SUBJ/$PROJNAME/input/

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/work/scripts
cp -ra $SCRIPTS/* /scratch/$USER/$SUBJ/$PROJNAME/work/scripts/
chmod +x /scratch/$USER/$SUBJ/$PROJNAME/work/scripts/*

mkdir -p /scratch/$USER/$SUBJ/$PROJNAME/output
cd /scratch/$USER

$SINGULARITY exec \
--bind "/scratch/${USER}/${SUBJ}/${PROJNAME}:/scratch" \
$IMAGE \
/scratch/work/scripts/startup_DiReCT_cortical_thickness.sh

mkdir -p $OUTPUT/$SUBJ
rsync -ra /scratch/$USER/$SUBJ/$PROJNAME/output/* $OUTPUT/$SUBJ/
rm -fr /scratch/$USER/$SUBJ/$PROJNAME
exit
