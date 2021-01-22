#!/bin/bash
#
#SBATCH --job-name=slurm-test
#SBATCH --output=test.txt
#SBATCH --nodes=1
#
#SBATCH --array=1-8

CWD=/home/carta/docker_repo
PY_MAIN=/home/carta/docker_repo/deep_lmn/laes_encode.py

ch-run --set-env=config.env --cd=$CWD antonio.lmn /opt/conda/bin/python $PY_MAIN -- --id=$SLURM_ARRAY_TASK_ID
