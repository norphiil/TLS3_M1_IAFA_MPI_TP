# python3 syracuse.py 4
import time
from mpi4py import MPI


def nb_primes(n: int) -> int:
    result: int = 0
    for i in range(1, n + 1):
        if n % i == 0:
            result += 1
    return result


def split_equal(x, n) -> list:
    result = [0]
    offset = 0
    if (x < n):
        return result
    elif (x % n == 0):
        for i in range(n):
            result.append(x // n + offset)
            offset += x // n
    else:
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if (i >= zp):
                result.append(pp + 1 + offset)
                offset += pp + 1
            else:
                result.append(pp + offset)
                offset += pp

    return result


N: int = 10000

comm = MPI.COMM_WORLD
rank: int = comm.rank
size: int = comm.size

if rank == 0:
    start_time = time.time()
    split: list = split_equal(N, size)
else:
    start_time = None
    split = None

split = comm.bcast(split, root=0)
current_max = 0
for val in range(split[rank] + 1, split[rank + 1] + 1):
    prime: int = nb_primes(val)
    current_max = max(current_max, prime)

max = comm.reduce(current_max, op=MPI.MAX, root=0)

if rank == 0:
    end_time = time.time()
    print(str(end_time - start_time) + " seconds")
    print(max)
