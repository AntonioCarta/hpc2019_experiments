"""
    benchmark script for RNN models using MKL-DNN.
"""
import time
import torch
import torch.nn as nn
import argparse


ALLOW_CUDA = False

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


def bench():
    # print(torch.__config__.parallel_info())
    # print(f"intra-op threads: {torch.get_num_threads()}")

    T, B, F = 100, 64, 300
    H = 200
    n_trials = 10
    fake_input = cuda_move(torch.zeros(T, B, F))

    # print(str(model), end='')
    rnn = cuda_move(nn.LSTM(F, H, num_layers=3, bidirectional=True))
    # rnn = script_lstm(F, H, num_layers=3, bidirectional=True)
    y = rnn(fake_input)

    def foo():
        rnn.zero_grad()
        y = rnn(fake_input)
        e = y[0].sum()
        e.backward()
        return e

    print(timeit_best(foo, n_trials))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', metavar='N', type=int, help='number of intra-op thread')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    bench()
