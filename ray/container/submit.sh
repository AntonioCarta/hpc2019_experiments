#!/bin/bash
#
#SBATCH --job-name=test_ray_charliecloud
#SBATCH --nodes=1
WORKDIR=/home/carta/hpc2019_experiments/ray/container
PY_MAIN=/home/carta/hpc2019_experiments/ray/container/demo.py

ch-run --set-env=config.env --cd=$WORKDIR antonio.ray /opt/conda/bin/python $PY_MAIN
