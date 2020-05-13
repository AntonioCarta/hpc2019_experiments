import time
import torch


ALLOW_CUDA = True  # Global variable to control cuda_move allocation behavior


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

