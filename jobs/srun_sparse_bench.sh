ENV=/home/carta/hpc2019_experiments/latest_image/config.env
CWD=/home/carta/hpc2019_experiments
CONTAINER=/home/carta/docker_repo/docker_latest/antonio.lmn
MAIN=/home/carta/hpc2019_experiments/run_bench_sweep_sparse.sh

srun --cpus-per-task=56 ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
