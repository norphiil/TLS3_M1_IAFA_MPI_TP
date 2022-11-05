# python3 syracuse.py 4
import math
import time
from mpi4py import MPI


def nb_primes(n: int) -> int:
    result: int = 0
    for i in range(1, n + 1):
        if n % i == 0:
            result += 1
    return result


N: int = 10000

comm = MPI.COMM_WORLD
rank: int = comm.rank
size: int = comm.size

if rank == 0:
    start_time = time.time()

current_max = 0
to: int = math.ceil(N / size)
for val in range(0, to):
    val = val * size + rank
    if val <= N:
        prime: int = nb_primes(val)
        current_max = max(current_max, prime)

max = comm.reduce(current_max, op=MPI.MAX, root=0)

if rank == 0:
    end_time = time.time()
    print(str(end_time - start_time) + " seconds")
    print(max)
