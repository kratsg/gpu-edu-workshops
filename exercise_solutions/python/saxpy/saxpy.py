
from numbapro import cuda
import numpy as np

@cuda.autojit
def saxpy(a, b, c):
    # Determine our unique global thread ID, so we know which element to process
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x;
    
    if ( tid < c.size ): # Make sure we don't do more work than we have data!
        c[tid] = 2 * a[tid] + b[tid];

def main():
    N = 2048 * 2048

    # Allocate host memory arrays
    a = np.empty(N)
    b = np.empty(N)
    c = np.empty(N)

    # Initialize host memory
    a.fill(2)
    b.fill(1)
    c.fill(0)

    # Allocate and copy GPU/device memory
    d_a = cuda.to_device(a)
    ## FIXME: allocate space for the other vectors ##
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 128
    number_of_blocks = N / threads_per_block + 1
## FIXME: given the threads_per_block is set, determine how many blocks we need ##

    saxpy [ number_of_blocks, threads_per_block ] ( d_a, d_b, d_c )
## FIXME: what variables do we pass? ##, N )

    ## FIXME: copy the vector c from the GPU back to the host ##
    d_c.copy_to_host( c )

    # Print out the first and last 5 values of c for a quality check
    print str(c[0:5])
    print str(c[-5:])
    
main() # Execute the program
