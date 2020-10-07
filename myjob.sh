sleep 5
uname -n
echo $SLURM_PROCID "Hello world!!!"
echo $SLURM_ARRAY_TASK_ID "Hello world!!!"
