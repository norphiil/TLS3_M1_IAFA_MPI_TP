#! /usr/bin/python3

import math
import time
import random

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
print("work on " + str(rank))
if __name__ == '__main__':
    nb = 100001
    inside = 0
    random.seed(rank)

    start_time = time.time()
    for i in range(math.floor(nb / size) * (rank), math.ceil(nb / size) * (rank + 1)):
        if (i < nb):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1:
                inside += 1
    end_time = time.time()

    inside_global = comm.reduce(inside, op=MPI.SUM, root=0)

    if (rank == 0):
        pi = 4 * inside_global / nb
        print(pi, str(end_time - start_time) + " seconds")
