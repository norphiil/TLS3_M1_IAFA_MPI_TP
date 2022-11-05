# hello.py
# mpirun -n 4 /usr/local/opt/python/libexec/bin/python hello.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
print("hello, il y a", size, "process parallèles")
