#!/bin/tcsh
setenv DATA # set datapath here, e.g. /autofs/cluster/iaslab/FSMAP/FSMAP_data/BIDS_modeled
setenv SCRIPTPATH # set script path here, e.g. /autofs/cluster/iaslab/users/jtheriault/FSMAP/scripts
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/jtnipyutil/jtnipyutil-2019-01-03-4cecb89cb1d9.simg # path to singularity image, built from docker pull jtheriaultpsych/jtnipyutil
setenv FWHM $1 # input from command line, see example_lvl2_model
setenv PROJNAME lvl2_tfce # unique project identifier
setenv SINGULARITY /usr/bin/singularity # location of singularity program on cluster.

mkdir -p /scratch/jq86/$PROJNAME/wrkdir/
mkdir -p /scratch/jq86/$PROJNAME/data
mkdir -p /scratch/jq86/$PROJNAME/output

rsync -ra $DATA/example/* /scratch/jq86/$PROJNAME/data

rsync $SCRIPTPATH/{lvl2_model.py,lvl2_model_startup.sh} /scratch/jq86/$PROJNAME/wrkdir/
chmod +x /scratch/jq86/$PROJNAME/wrkdir/lvl2_model_startup.sh
cd /scratch/jq86

$SINGULARITY exec  \
--bind "/scratch/jq86/$PROJNAME/data:/scratch/data" \
--bind "/scratch/jq86/$PROJNAME/output:/scratch/output" \
--bind "/scratch/jq86/$PROJNAME/wrkdir/:/scratch/wrkdir" \
--bind "/homes/9/jq86:/license" \
$IMAGE \
/scratch/wrkdir/lvl2_model_startup.sh

mkdir -p $DATA/lvl2/$PROJNAME
cp -r /scratch/jq86/$PROJNAME/output/* $DATA/lvl2/$PROJNAME
rm -r /scratch/jq86/$PROJNAME
exit

# I was running into a problem because of the tcsh shell on the MGH cluster.
# Getting this error after running $SINGULARITY shell $IMAGE:
#
# Singularity: Invoking an interactive shell within container...
# ERROR: Shell does not exist in container: /usr/local/bin/tcsh
# ERROR: Using /bin/sh instead...
#
# Solution was to create a startup shell script, start_mdl-smooth-prewhite.#!/bin/sh
# This runs :
# /neurodocker/startup.sh \
# python /scratch/wrkdir/mdl_smooth-prewhite.py $SUBJ

# The \ after /neurodocker/startup.sh is necessary, as the python command runs inside a new image invoked.
# The neurodocker/startup.sh command activates the python environment necessary to run all the installed packages.

# This will be important for using the Singularity build in the future.
