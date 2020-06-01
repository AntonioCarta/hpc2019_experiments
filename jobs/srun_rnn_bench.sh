ENV=/home/carta/docker_repo/docker_latest/config.env
CWD=/home/carta/docker_repo
CONTAINER=/home/carta/docker_repo/docker_latest/antonio.lmn
MAIN=/home/carta/hpc2019_experiments/run_bench_sweep_rnn.sh
srun --cpus-per-task=56 ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
