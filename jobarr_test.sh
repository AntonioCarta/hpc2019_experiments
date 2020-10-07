#!/bin/bash
#
#SBATCH --oversubscribe
#SBATCH --job-name=slurm_test
#SBATCH --partition=xeon
#SBATCH --output=./slurm_logs/job%j.out
#SBATCH --array=1-128
#SBATCH --distribution=block:block
#SBATCH --ntasks=1
#SBATCH --mem=1024K
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=4
#SBATCH --cores-per-socket=8

ENV=/home/carta/hpc2019_experiments/latest_image/config.env
CWD=/home/carta/cl_code/recurrent_continual_learning
CONTAINER=/home/carta/hpc2019_experiments/latest_image/antonio.latest
MAIN=/home/carta/hpc2019_experiments/myjob.sh
# srun --mem=1024K --oversubscribe --ntasks 1 --cpus-per-task 1 --cpu_bind=cores --distribution=block:block --output=job%j-%t.out ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
# ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN -- --id="$SLURM_ARRAY_TASK_ID"
