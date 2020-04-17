#!/bin/bash
#
#SBATCH --job-name=/home/carta/docker_repo/docker_latest/test
#SBATCH --output=test.txt
#SBATCH --nodes=1
#
#SBATCH --array=1-8

ch-run --set-env=config.env --cd=/home/carta/docker_repo antonio.lmn /opt/conda/bin/python /home/carta/docker_repo/deep_lmn/laes_encode.py -- --id=$SLURM_ARRAY_TASK_ID
