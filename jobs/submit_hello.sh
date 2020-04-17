#!/bin/bash
#
#SBATCH --job-name=slurm_test
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024K
#SBATCH --array=1-8
WORKDIR=/home/carta/docker_repo/jobs
CONTAINER=../docker_latest/antonio.lmn
ch-run --cd=$WORKDIR $CONTAINER ./hello.sh -- $SLURM_ARRAY_TASK_ID
