#!/bin/bash
#
#SBATCH --job-name=cl_cosine_lstm
#SBATCH --partition=debug
#SBATCH --cores-per-socket=8
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=4
#SBATCH --mem=1024K
#SBATCH --nodelist=c01
#SBATCH --ntasks=1
#SBATCH --output=./slurm_logs/slurm/job%j.out

ENV=/home/carta/hpc2019_experiments/latest_image/config.env
CWD=/home/carta/cl_code/recurrent_continual_learning
CONTAINER=/home/carta/hpc2019_experiments/latest_image/antonio.latest
MAIN=/home/carta/hpc2019_experiments/myjob.sh
srun --ntasks 128 --cpus-per-task 8 --cpu_bind=cores --distribution=block:block --output=job%j-%t.out ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
# ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN -- --id="$SLURM_ARRAY_TASK_ID"
