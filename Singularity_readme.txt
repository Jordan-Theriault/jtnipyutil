** THIS NO LONGER WORKS. WAITING FOR A BUG FIX. SEE https://github.com/docker/hub-feedback/issues/727

To create a singularity image, ensure that under Docker Daemon commands, the following is entered:
{
  "debug" : true,
  "storage-driver" : "aufs",
  "experimental" : true
}

The storage driver must be "aufs", or else you will run into errors unpacking .tar files when converting to Singularity using singularityware/docker2singularity

Another potential option was to ensure that tar commands were replaced with bsdtar commands (see neurodocker command).
e.g.
--run "export tar='bsdtar'" \
But this behaved inconsistently with Dockerhub.

As of now, to convert to singularity, I need to build the image on my Desktop, then convert to singularity using singularityware/docker2singularity

e.g.
from /jtnipyutil:
docker build -t jtnipyutil .
docker run --privileged -t --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /Users/jtheria/singularity_images:/output \
    singularityware/docker2singularity \
    jtnipyutil
