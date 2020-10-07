# Risultati jemalloc e training distribuito
test singolo nodo su hpc2019 (ho usato c01).

## default
ottenuti con `./run_basic.sh`
```
### using KMP_AFFINITY=granularity=fine,compact,1,0
### using OMP_NUM_THREADS=1
1.6808412075042725
### using OMP_NUM_THREADS=2
1.1272108554840088
### using OMP_NUM_THREADS=4
0.8186666965484619
### using OMP_NUM_THREADS=7
0.6703438758850098
### using OMP_NUM_THREADS=14
0.7013509273529053
### using OMP_NUM_THREADS=28
0.7092685699462891
### using OMP_NUM_THREADS=56
0.7323026657104492
### using OMP_NUM_THREADS=112
0.7249910831451416
```

## jemalloc
ottenuti con `./run_fast.sh`
```
1.6570978164672852
### using OMP_NUM_THREADS=2
1.0977656841278076
### using OMP_NUM_THREADS=4
0.8578088283538818
### using OMP_NUM_THREADS=7
0.7062547206878662
### using OMP_NUM_THREADS=14
0.6608099937438965
### using OMP_NUM_THREADS=28
0.6939985752105713
### using OMP_NUM_THREADS=56
0.6995651721954346
### using OMP_NUM_THREADS=112
0.6940324306488037
```

## distributed
ottenuti con `./run_distributed.sh`
world_size = 2
overhead di parecchi secondi per il setup, escluso dai conti.

```
### using KMP_AFFINITY=granularity=fine,compact,1,0
### using OMP_NUM_THREADS=14
### using s0 prefix: numactl -C0-13 -m0
### using s1 prefix: numactl -C14-27 -m1
Namespace(rank=0, threads=None, world_size=2)
Namespace(rank=1, threads=None, world_size=2)
0.927436113357544
Running basic DDP example on rank 0.
tensor(-3686.3323, grad_fn=<SumBackward0>)
0.9242987632751465
Running basic DDP example on rank 1.
tensor(-3686.3323, grad_fn=<SumBackward0>)
```
