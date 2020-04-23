ENV=../docker_latest/config.env
CWD=/home/carta/docker_repo
CONTAINER=../docker_latest/antonio.lmn
MAIN=/home/carta/docker_repo/run_bench_docker.sh
srun --cpus-per-task=56 ch-run --set-env=$ENV --cd=$CWD $CONTAINER $MAIN
