#!/bin/bash

#SBATCH --job-name=test
#SBATCH --cpus-per-task=5
#SBATCH --nodes=3
#SBATCH --tasks-per-node 1

worker_num=2 # Must be one less that the total number of nodes
WORKDIR=/home/carta/hpc2019_experiments/ray/container
PY_MAIN=/home/carta/hpc2019_experiments/ray/container/demo_multinode.py

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
export redis_password

srun --nodes=1 --ntasks=1 -w $node1 ch-run --set-env=config.env antonio.ray ray -- start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ch-run --set-env=config.env antonio.ray ray -- start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

ch-run --set-env=config.env --cd=$WORKDIR antonio.ray python $PY_MAIN