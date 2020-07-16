# How to Run Containers on the Intel Cluster

This repository describes my personal experience porting a python project from a single node to the Intel cluster `hpc2019.unipi.it`.

The use case presented here is a simple example of parallel computation. The python script that we want to run implements the training of a deep learning model, and we would like to run a model selection in parallel, with several instances of the same script running on different compute nodes of the cluster, each one with different parameters. This approach does not require communication between the jobs or communication between compute nodes since each configuration runs on a single node. If you are interested in multi-node training, you can look up horovod, a deep learning library that allows you to easily scale to multiple nodes.

In order to run the python script, the operating system must have a set of scientific libraries installed. Different users may require different version of the libraries, possibly incompatible between them. Since installing and managing all the dependencies on the cluster machines would be unfeasible, we will encapsulate the dependencies inside a Charliecloud container, which is a containerization mechanism designed for HPC systems.
    
## Outline
The process of porting the experiment is articulated in 4 steps:
- export of your python environment
- creation of a charliecloud image
- creation of a slurm submission script
- scheduling of parallel jobs on the cluster with slurm

# Creation of a Charliecloud Container
To create a container to run you python scripts, you must export your python environment. This will allow to easily install all the dependencies inside a container. You can use the command:

```
    conda list -e > requirements.txt
```

Using the requirements, we can create a Dockerfile, which will be used to create our container and install the relevant dependencies. The `Dockerfile` is based on the Anaconda Docker image.

```
    FROM continuumio/miniconda3
    ADD requirements.txt /requirements.txt
    RUN conda install python=3.8
    RUN conda install --file requirements.txt
    RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    RUN pip install fbpca
    WORKDIR /home/carta
```

Notice that despite having all the environment saved in the \verb!requirements.txt! file, I had some problems during the installation and therefore I had to install pytorch separately after the other libraries.

The next step is the creation of the Charliecloud container from the Dockerfile. In my case, the container is called `antonio/base`
```
    ch-build -t antonio/lmn --network=host .
    ch-builder2tar antonio/lmn .
    ch-tar2dir ./antonio.lmn.tar.gz .        
```

A script to build the charliecloud image is available in `create_image.sh` that you can run with the command:
```
create_image IMAGE_NAME IMAGE_FOLDER
```

For some unknown reasons, Charliecloud ignores the ENV directives defined in your Dockerfile. If you want to define additional environment variables for your container you can create an environment file that defines your custom variables. This file must be passed as an argument when running the container, as we will see below. In my case, I added to the `PYTHONPATH` the folder where my python library resides in a separate `config.env` file:
```
    PYTHONPATH=.:/home/carta:/home/carta/cannon
```

Now we can easily run our python script with the command
```
    ch-run --set-env=config.env 
        --cd=/home/carta/hpc2019_experiments antonio.lmn 
        /opt/conda/bin/python /home/carta/repo/main.py
```

where:
- `--cd` changes the directory
- `--set-env` set the environment variables

The container executed with `ch-run` is running on the head node.

# Scheduling Jobs on the Cluster
Since we do not have direct access to the compute nodes, we must use \verb!slurm! to schedule the jobs in parallel. `slurm` will take care of the allocation of the resources in a fair way for all the jobs.

To do this, we need to create a submission script `submit.sh`, which will be run by slurm:

```
    #!/bin/bash
    #
    #SBATCH --job-name=/home/carta/docker_repo/docker_latest/test
    #SBATCH --output=test.txt
    #SBATCH --nodes=1
    #
    #SBATCH --array=1-8

    ch-run --set-env=config.env 
        --cd=/home/carta/docker_repo antonio.lmn 
        /opt/conda/bin/python /home/carta/repo/main.py 
        -- --id=$SLURM_ARRAY_TASK_ID
```

There are a couple of things to notice about the submission script:
- Comments prefixed by `#SBATCH` are used to pass options to the `sbatch` command. You can read more about `sbatch` options on the manual.
- The use of job arrays to easily manage all the jobs together. This submission script schedule 8 jobs (`--array=1-8`).
- Specification of the resources for each job (`--nodes=1`).
- The argument `$SLURM_ARRAY_TASK_ID`, set by slurm to match the job id. In this case it is directly passed to the python script, which will use it to change the model's configuration.


To submit the jobs to the scheduler, you must run the command
```
    sbatch submit.sh
```

You can check the status of your job with `squeue`, where you should see a list of your jobs in execution.


# Distributed Execution with Ray
Ray is a python library that provides simple primitives for running distributed applications. Many libraries are built on top of Ray for traditional workloads such as model selection (Ray Tune) or reinforcement learning (RLLib). Ray can be used on the cluster with charliecloud, as shown in the examples available in `ray/container`:
- `Dockerfile` installs ray on a container.
- `submit.sh` creates the charliecloud image from the `Dockerfile`
- `submit_multinode.sh` is the submission script for multinode experiments.
- `demo_multinode.py` is the Python script executed with `submit_multinode.sh` that shows basic Ray functionality.

## Submission Script
The submission script in the examples directory performs the following steps:
- start ray head node
- start each worker separately
- start the main script

You have to make only a couple of minimal changes to adapt the script to your job:
- change `job-name` to give it a significative name.
- set `#SBATCH --nodes` slurm argument to the number of desired nodes + 1 for the head node.
- set `worker_num` to the number of worker nodes (which must be `#SBATCH --nodes` - 1).
- set `WORKDIR` and `PY_MAIN` to the working directory and main script, respectively.

```
#!/bin/bash

#SBATCH --job-name=test
#SBATCH --cpus-per-task=5
#SBATCH --nodes=3
#SBATCH --tasks-per-node 1

worker_num=2 # Must be one less that the total number of nodes
WORKDIR=/home/carta/hpc2019_experiments/ray/container
PY_MAIN=/home/carta/hpc2019_experiments/ray/container/demo_multinode.py

...
```

You can run the submission script with the command `sbatch submit_multinode.sh`.

## Main Script

Inside the main script, Ray must be initialized with the IP address of the head node and the redis password, which are saved into appropriate environment variables by the submission script:
```
    ray.init(address=os.environ["ip_head"], redis_password=os.environ["redis_password"])
```

Remote functions are defined by adding a `ray.remote` decorator to the function's definition. `ray.remote` takes as additional values the number of requested resources (`num_cpus`, `num_gpus`). However, Ray does not enforce exclusive access to the resources, i.e. each remote function can use all the cores available to its worker node. The programmer is responsible for the resource management.

```
@ray.remote(num_cpus=2)
def f():
    print(f"Inside a remote function")
    time.sleep(5)
    return ray.services.get_node_ip_address()
```

Remote function can be called asynchronously with the `remote` method, which immediately returns a handle (a Future value):
```
remote_id = f.remote() 
```

The return value can be obtained with the `ray.get` method, which waits until the remote function returns a value.
```
ray.get(remote_id)
```

Resource management in pytorch can be done with the methods `torch.get_num_threads()` and `torch.set_num_threads()`. They must be called before any computation. Since ray does not enforce resource management, these methods are fundamental to ensure that each remote function is using the correct amount of resources.

# Troubleshooting
If you have any problems you can:
- Try to run without containers to check if charliecloud is causing the issue.
- Check the number of used cores with `ssh c03` and then running `htop` inside the worker node, where `c03` is the node you are using (you can find it with `squeue`).
- Check the number of pytorch threads with `torch.get_num_threads()` and `torch.set_num_threads()`.

# Final Notes
I hope you will find this helpful to start running your own experiments on the cluster. While this document describes my personal experience, using the steps described here and the official documentation it should be easy enough to adapt the submission scripts to your own use case.

# Useful Links
These are some resources that you may find useful:

- **Docker docs** https://docs.docker.com/
- **Charliecloud docs** https://hpc.github.io/charliecloud/index.html
- **slurm docs** https://slurm.schedmd.com/documentation.html
- **Ray docs** https://docs.ray.io/en/latest/