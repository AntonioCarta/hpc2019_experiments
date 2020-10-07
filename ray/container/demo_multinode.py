from collections import Counter
import os
import sys
import time
import ray
import torch
import multiprocessing

ray.init(address=os.environ["ip_head"], redis_password=os.environ["redis_password"])

print("Ray nodes information:")
print(ray.nodes())

print(f"pytorch threads: {torch.get_num_threads()}")

@ray.remote(num_cpus=2)
def f():
    print(f"multiprocessing cores: {multiprocessing.cpu_count()}")
    print(f"pytorch threads (before): {torch.get_num_threads()}")
    torch.set_num_threads(2)
    print(f"pytorch threads (after): {torch.get_num_threads()}")
    time.sleep(5)
    return ray.services.get_node_ip_address()

# Each iteration takes 50 second (assuming that ray was able to access all of the allocated nodes).
for i in range(10):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(56)])  # each node on hpc2019 has 56 cores.
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)

print("Done.")
