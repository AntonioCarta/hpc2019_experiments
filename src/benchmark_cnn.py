"""
    benchmark script for RNN models using MKL-DNN.
"""
# from src.utils import timeit_best, set_allow_cuda, cuda_move
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time


ALLOW_CUDA = True

def set_allow_cuda(b):
    global ALLOW_CUDA
    ALLOW_CUDA = b
    if b:
        print("CUDA enabled.")
    else:
        print("CUDA disabled.")


def set_gpu():
    import os
    try:
        import gpustat
    except ImportError as e:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]

        print("Setting GPU to: {}".format(bestGPU))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    except BaseException as e:
        print("GPU not available: " + str(e))


def cuda_move(args):
    """ Move a sequence of tensors to CUDA if the system supports it. """
    if not ALLOW_CUDA:
        return args.cpu()
    b = torch.cuda.is_available()
    # for t in args:
    #     if b:
    #         yield t.cuda()
    #     else:
    #         yield t
    if b:
        return args.cuda()
    else:
        return args


def timeit(foo, n_trials):
    times = []
    for _ in range(n_trials):
        start = time.time()
        foo()
        end = time.time()
        t = end - start
        times.append(t)
    t_min = min(times)
    t_mean = sum(times) / n_trials
    print("min: {:.5f}, mean: {:.5f}, n_trials: {}".format(t_min, t_mean, n_trials))


def timeit_best(foo, n_trials):
    times = []
    for _ in range(n_trials):
        start = time.time()
        foo()
        end = time.time()
        t = end - start
        times.append(t)
    return min(times)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def bench():
    # print(torch.__config__.parallel_info())
    # print(f"intra-op threads: {torch.get_num_threads()}")
    fake_input = cuda_move(torch.zeros(100, 3, 32, 32))
    n_trials = 10

    net = cuda_move(Net())
    y = net(fake_input)

    def foo():
        net.zero_grad()
        y = net(fake_input)
        e = y[0].sum()
        e.backward()
        return e

    print(timeit_best(foo, n_trials))


if __name__ == '__main__':
    # set_allow_cuda(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', metavar='N', type=int, help='number of intra-op thread')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    bench()
