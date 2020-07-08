from collections import Counter
import os
import sys
import time
import ray

ray.init()


@ray.remote
def f():
    time.sleep(1)
    return ray.services.get_node_ip_address()

# The following takes one second (assuming that ray was able to access all of the allocated nodes).
for i in range(10):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(8)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)

print("Done.")
