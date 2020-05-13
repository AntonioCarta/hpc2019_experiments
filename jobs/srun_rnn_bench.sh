ENV=../latest_image/config.env
CWD=/home/carta/docker_repo
CONTAINER=../latest_image/antonio.lmn
MAIN=/home/carta/docker_repo/run_bench_sweep.sh
srun --cpus-per-task=56 ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
