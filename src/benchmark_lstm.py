"""
    benchmark script for RNN models using MKL-DNN.
"""
from cannon.utils import set_allow_cuda, timeit_best
import torch
import torch.nn as nn
import argparse
from torch import jit
from typing import List
from torch import Tensor
# from src.custom_lstm import script_lstm

def bench():
    # print(torch.__config__.parallel_info())
    # print(f"intra-op threads: {torch.get_num_threads()}")

    T, B, F = 100, 64, 300
    H = 200
    n_trials = 10
    fake_input = torch.zeros(T, B, F)

    # print(str(model), end='')
    rnn = nn.LSTM(F, H, num_layers=3, bidirectional=True)
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

    # models = [JitRNN, SlowRNN]
    bench()
