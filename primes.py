# python3 syracuse.py 4
import sys


def nb_primes(n):
    result = 0
    for i in range(1, n + 1):
        if n % i == 0:
            result += 1
    return result


N: int = 10000

current_max = 0

for val in range(1, N + 1):
    tmp = nb_primes(val)
    current_max = max(current_max, tmp)

print(current_max)
