# hello.py
# mpirun -n 4 /usr/local/opt/python/libexec/bin/python hello.py
from mpi4py import MPI
import platform
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("hello world from process ", rank, "working on ", platform.node())
