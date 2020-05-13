"""
    benchmark script for RNN models using MKL-DNN.
"""
from src.utils import set_allow_cuda, cuda_move, timeit_best
import torch
from torch import nn
from torch import jit
from torch import Tensor
import argparse


class JitRNN(jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, in_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p)

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.Wxh.device)

    @jit.script_method
    def forward(self, x):
        assert len(x.shape) == 3
        out = []
        h_curr = self.init_hidden(x.shape[1])
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_curr = torch.tanh(torch.mm(xt, self.Wxh.t()) + torch.mm(h_curr, self.Whh.t()) + self.bh)
            out.append(h_curr)
        return torch.stack(out), h_curr


class SlowRNN(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, in_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.Wxh.device)

    def forward(self, x):
        assert len(x.shape) == 3
        out = []
        h_curr = self.init_hidden(x.shape[1])
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_curr = torch.tanh(torch.mm(xt, self.Wxh.t()) + torch.mm(h_curr, self.Whh.t()) + self.bh)
            out.append(h_curr)
        return torch.stack(out), h_curr


def bench(model):
    # print(torch.__config__.parallel_info())
    # print(f"intra-op threads: {torch.get_num_threads()}")

    T, B, F = 100, 64, 300
    H = 200
    n_trials = 10
    fake_input = cuda_move(torch.randn(T, B, F))

    def foo_step(m, x):
        y = m(x)[0]
        e = y.sum()
        e.backward()

    # print(str(model), end='')
    rnn = cuda_move(model(F, H))
    y = rnn(fake_input)
    foo = lambda: foo_step(rnn, fake_input)
    print(timeit_best(foo, n_trials))
    print(fake_input.device)


if __name__ == '__main__':
    set_allow_cuda(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', metavar='N', type=int, help='number of intra-op thread')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # models = [JitRNN, SlowRNN]
    bench(SlowRNN)
