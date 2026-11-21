"""Settings of the tests, before pollux is imported."""
import os

# few threads: numba and OpenMP otherwise start one thread per core of the node (hundreds on a login node), which
# makes each call of a parallel kernel slower than the work itself on these small inputs
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
