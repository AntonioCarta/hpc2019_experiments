import os
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import torch
import torch.nn as nn
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
ALLOW_CUDA = False


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def timeit_best(foo, n_trials):
    times = []
    for _ in range(n_trials):
        start = time.time()
        foo()
        end = time.time()
        t = end - start
        times.append(t)
    return min(times)


def set_allow_cuda(b):
    global ALLOW_CUDA
    ALLOW_CUDA = b
    if b:
        print("CUDA enabled.")
    else:
        print("CUDA disabled.")


def cuda_move(args):
    """ Move a sequence of tensors to CUDA if the system supports it. """
    if not ALLOW_CUDA:
        return args.cpu()
    b = torch.cuda.is_available()
    if b:
        return args.cuda()
    else:
        return args


def bench(rank, world_size):
    setup(rank, world_size)

    T, B, F = 100, 64, 300
    H = 200
    n_trials = 10
    fake_input = cuda_move(torch.zeros(T, B, F))

    rnn = DDP(cuda_move(nn.LSTM(F, H, num_layers=3, bidirectional=True)))
    y = rnn(fake_input)

    def foo():
        rnn.zero_grad()
        y = rnn(fake_input)
        e = y[0].sum()
        e.backward()
        return e

    print(timeit_best(foo, n_trials))
    print(f"Running basic DDP example on rank {rank}.")
    print(y[0].sum())
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', metavar='N', type=int, help='number of intra-op thread')
    parser.add_argument('--rank', metavar='N', type=int)
    parser.add_argument('--world_size', metavar='N', type=int)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print(args)

    # run_demo(bench, 2)
    bench(args.rank, args.world_size)