import sys
sys.path.append('build/lib.linux-x86_64-3.10')

from mpi4py import MPI
import _s4m
import time
import random

def test(s4m_comm, mpi_comm, data_size):
    steps = 10
    mpi_comm.Barrier()
    num_procs = mpi_comm.Get_size()
    my_rank = mpi_comm.Get_rank()
    random.seed(my_rank)
    expected_msg = (num_procs-1)*steps
    data = bytes(data_size)
    for i in range(0, steps):
        s = 5 + random.randint(0,5)
        time.sleep(s)
        t1 = time.time()
        s4m_comm.post(data)
        t2 = time.time()
        print(f'[{my_rank}] post took {(t2-t1):.4f}')
        sys.stdout.flush()
        t1 = time.time()
        h = s4m_comm.get()
        t2 = time.time()
        print(f'[{my_rank}] get took {(t2-t1):.4f}')
        sys.stdout.flush()
        if h is not None:
            expected_msg -= 1
    for i in range(0, expected_msg):
        t1 = time.time()
        s4m_comm.get(blocking=True)
        t2 = time.time()
        print(f'[{my_rank}] get(blocking) took {(t2-t1):.4f}')
        sys.stdout.flush()


if __name__ == '__main__':
    s4m_comm = _s4m.S4MCommunicator(MPI.COMM_WORLD, "ofi+tcp")
    mpi_comm = MPI.COMM_WORLD
    test(s4m_comm, mpi_comm, 10)
