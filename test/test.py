import sys
from mpi4py import MPI
import s4m
import time
import random

def test(s4m_service, mpi_comm, data_size):
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
        # broadcast takes any class that satisfies the buffer
        # protocol, e.g. bytes, str, or numpy, as long as the
        # underlying buffer is contiguous. If you need to send
        # something else, a good way is to pickle the data into
        # a bytes buffer.
        s4m_service.broadcast(data)
        t2 = time.time()
        print(f'[{my_rank}] broadcast took {(t2-t1):.4f}')
        sys.stdout.flush()
        t1 = time.time()
        # The receive function is non-blocking and will check
        # for available data sent by other processes. If data
        # is available, the function will return a pair (source, data)
        # where source is the rank that sent the data, and data is a
        # bytes object. If no data is available, the function will
        # return None.
        h = s4m_service.receive()
        t2 = time.time()
        print(f'[{my_rank}] receive took {(t2-t1):.4f}')
        sys.stdout.flush()
        if h is not None:
            expected_msg -= 1
    for i in range(0, expected_msg):
        t1 = time.time()
        # blocking=True can be provided to make the receive
        # function block until data is available.
        s4m_service.receive(blocking=True)
        t2 = time.time()
        print(f'[{my_rank}] receive(blocking) took {(t2-t1):.4f}')
        sys.stdout.flush()
    # We need to prevent the destructor of the S4MService
    # from being called when other processes could still be
    # sending data, hence the barrier.
    mpi_comm.Barrier()


if __name__ == '__main__':
    # The constructor is going to do some collective communication
    # across processes of the provided MPI communicator, so make
    # sure this call is done by all the processes at the same time.
    s4m_service = s4m.S4MService(MPI.COMM_WORLD, "ofi+tcp")
    mpi_comm = MPI.COMM_WORLD
    test(s4m_service, mpi_comm, 10)
