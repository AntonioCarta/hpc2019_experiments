"""
    benchmark script for RNN models using MKL-DNN.
"""
from src.utils import set_allow_cuda, cuda_move, timeit_best
import torch
from torch import nn
import argparse


class SparseModule(nn.Module):
    def __init__(self, n_items, input_size, output_size):
        super().__init__()
        idx_xs = torch.randint(0, input_size, (n_items,))
        idx_ys = torch.randint(0, output_size, (n_items,))
        idxs = torch.stack([idx_ys, idx_xs], dim=0)
        vals = torch.randn(n_items)
        self.W = nn.Parameter(torch.sparse.FloatTensor(idxs, vals, (output_size, input_size)))

    def forward(self, x):
        assert len(x.shape) == 2
        x = x.transpose(1, 0)
        x = torch.sparse.mm(self.W, x)
        x = x.transpose(1, 0)
        return x


def debug_sparsemodule():
    x = torch.randn(32, 10)
    net = SparseModule(20, 10, 7)
    y = net(x)
    print(x.shape, y.shape)
    print(net.W.to_dense())


def bench():
    in_size, h_size, out_size = 10, 500, 7
    sparse_vals = 100
    batch_size = 128

    sparse_net = nn.Sequential(
        SparseModule(sparse_vals, in_size, h_size),
        nn.ReLU(),
        SparseModule(sparse_vals, h_size, h_size),
        nn.ReLU(),
        SparseModule(sparse_vals, h_size, h_size),
        nn.ReLU(),
        SparseModule(sparse_vals, h_size, out_size),
    )
    sparse_net = cuda_move(sparse_net)

    x = cuda_move(torch.randn(batch_size, in_size))
    y = sparse_net(x)

    n_trials = 10
    def foo_step(m, x):
        for _ in range(100):
            y = m(x)[0]
            e = y.sum()
            e.backward()

    y = sparse_net(x)
    foo = lambda: foo_step(sparse_net, x)
    print(timeit_best(foo, n_trials))
    # print(x.device)


if __name__ == '__main__':
    # set_allow_cuda(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', metavar='N', type=int, help='number of intra-op thread')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # models = [JitRNN, SlowRNN]
    bench()
