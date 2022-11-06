# version sequentielle
# python3 n-bodies.py 12 1000

# version parallele
# mpirun -n 3 python3 n-bodies.py 12 1000

from mpi4py import MPI
import sys
import math
import random
import matplotlib.pyplot as plt
import time
import numpy as np


def split(x, size):  # split a vector "x" in "size" part. In case it does not divide well, the last one receives one less than others
    n = math.ceil(len(x) / size)
    return [x[n * i:n * (i + 1)] for i in range(size - 1)] + [x[n * (size - 1):len(x)]]


def unsplit(x):  # unsplit a list x composed of lists
    y = []
    n = len(x)
    for i in range(n):
        for j in range(len(x[i])):
            y.append(x[i][j])
    return y


solarmass = 1.98892e30


def circlev(rx, ry):
    r2 = math.sqrt(rx * rx + ry * ry)
    numerator = (6.67e-11) * 1e6 * solarmass
    return math.sqrt(numerator / r2)


class Data_item:  # from http://physics.princeton.edu/~fpretori/Nbody/code.htm

    def __init__(self, id, positionx, positiony, speedx, speedy, weight):
        self.id = id
        self.positionx = positionx
        self.positiony = positiony
        self.weight = weight

        # the center of the world, very heavy one...
        if positionx == 0 and positiony == 0:
            self.speedx = 0
            self.speedy = 0
        else:
            if speedx == 0 and speedy == 0:  # initial values
                magv = circlev(positionx, positiony)
                absangle = math.atan(math.fabs(positiony / positionx))
                thetav = math.pi / 2 - absangle
                phiv = random.uniform(0, 1) * math.pi
                self.speedx = -1 * \
                    math.copysign(1, positiony) * math.cos(thetav) * magv
                self.speedy = math.copysign(
                    1, positionx) * math.sin(thetav) * magv
                # Orient a random 2D circular orbit
                if (random.uniform(0, 1) <= .5):
                    self.speedx = -self.speedx
                    self.speedy = -self.speedy
            else:
                self.speedx = speedx
                self.speedy = speedy

    def __str__(self):
        return "ID=" + str(self.id) + " POS=(" + str(self.positionx) + "," + str(self.positiony) + ") SPEED=(" + str(self.speedx) + "," + str(self.speedy) + ") WEIGHT=" + str(self.weight)


def display(m: str, list: list, rank: int = 0):
    for i in range(len(list)):
        print("PROC" + str(rank) + ":" + m + "-" + str(list[i]))


def displayPlot(d: list[Data_item]):
    plt.gcf().clear()			# to remove to see the traces of the particules...
    plt.axis((-1e17, 1e17, -1e17, 1e17))
    xx: list = [d[i].positionx for i in range(len(d))]
    yy: list = [d[i].positiony for i in range(len(d))]
    plt.plot(xx, yy, 'ro')
    plt.draw()
    plt.pause(0.00001)			# in order to see something otherwise too fast...


def interaction(i: Data_item, j: Data_item):
    dist = math.sqrt((j.positionx - i.positionx) * (j.positionx - i.positionx) +
                     (j.positiony - i.positiony) * (j.positiony - i.positiony))
    if i == j:
        return (0, 0)
    g = 6.673e-11
    factor = g * i.weight * j.weight / (dist * dist + 3e4 * 3e4)
    return factor * (j.positionx - i.positionx) / dist, factor * (j.positiony - i.positiony) / dist


def update(d: Data_item, f: list) -> Data_item:
    dt = 1e11
    vx = d.speedx + dt * f[0] / d.weight
    vy = d.speedy + dt * f[1] / d.weight
    px = d.positionx + dt * vx
    py = d.positiony + dt * vy
    return Data_item(id=d.id, positionx=px, positiony=py, speedx=vx, speedy=vy, weight=d.weight)


def signature(world: list[Data_item]):
    s = 0
    for d in world:
        s += d.positionx + d.positiony
    return s


def init_world(n: int) -> list[Data_item]:
    data = [Data_item(id=i, positionx=1e18 * math.exp(-1.8) * (.5 - random.uniform(0, 1)), positiony=1e18 * math.exp(-1.8) *
                      (.5 - random.uniform(0, 1)), speedx=0, speedy=0, weight=(random.uniform(0, 1) * solarmass * 10 + 1e20)) for i in range(n - 1)]
    data.append(Data_item(id=nbbodies - 1, positionx=0, positiony=0,
                speedx=0, speedy=0, weight=1e6 * solarmass))
    return data


def sub(tup1: tuple, tup2: tuple) -> tuple:
    return tuple(map(lambda i, j: i - j, tup1, tup2))


def add(tup1: tuple, tup2: tuple) -> tuple:
    return tuple(map(lambda i, j: i + j, tup1, tup2))


def split_equal(x, n) -> list:
    result = [0]
    offset = 0
    if (x < n):
        result.append(x)
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


def list_tuple_sum(x, y, dt=None):
    lenght = len(x)
    for i in range(0, lenght):
        y[i] = add(x[i], y[i])
    return y


def list_tuple_sub(x, y, dt=None):
    lenght = len(x)
    for i in range(0, lenght):
        y[i] = sub(x[i], y[i])
    return y


op_sum = MPI.Op.Create(list_tuple_sum)
op_sub = MPI.Op.Create(list_tuple_sub)


nbbodies = int(sys.argv[1])
NBSTEPS = int(sys.argv[2])
# une pause de 2 secondes, juste pour voir que ça s'affiche bien
# on doit l'enlever dès que ça marche ;)
# plt.pause(2)

comm = MPI.COMM_WORLD
rank: int = comm.rank
size: int = comm.size

# à modifier si on veut que le monde créé soit différent à chaque fois
random.seed(rank)

if rank == 0:
    plt.draw()
    plt.show(block=False)
    data = init_world(nbbodies)
    split_nbbodies = split_equal(nbbodies, size)
else:
    data = None
    split_nbbodies = None
data = comm.bcast(data, root=0)
split_nbbodies = comm.bcast(split_nbbodies, root=0)
# here to start the code...

saved_signature = []
if rank == 0:
    start_time = MPI.Wtime()
for t in range(0, NBSTEPS):

    force = [(0, 0)
             for _ in range(nbbodies)]
    force_j_i = [([(0, 0)
                  for _ in range(nbbodies)])
                 for _ in range(nbbodies)]
    for i in range(split_nbbodies[rank], split_nbbodies[rank + 1]):
        for j in range(i):
            force_j_i[i][j] = interaction(data[i], data[j])
            force[i] = add(force[i], force_j_i[i][j])

    force = comm.reduce(force, op=op_sum, root=0)

    force_sub = [(0, 0)
                 for _ in range(nbbodies)]
    for i in range(split_nbbodies[rank], split_nbbodies[rank + 1]):
        for j in range(i):
            force_sub[j] = add(force_sub[j], force_j_i[i][j])

    force_sub = comm.reduce(force_sub, op=op_sum, root=0)

    if rank == 0:
        force = list_tuple_sub(force, force_sub)
        # print('force: ' + str(force))
        for i in range(0, nbbodies):
            data[i] = update(data[i], force[i])
        displayPlot(data)
        saved_signature.append(signature(data))
    data = comm.bcast(data, root=0)

if rank == 0:
    end_time = MPI.Wtime()
    print('saved_signature: ' + str(saved_signature))
    print('saved_time: ' + str(end_time - start_time) + ' Secondes')
