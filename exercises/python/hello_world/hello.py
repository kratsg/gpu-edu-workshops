
from numbapro import cuda # Import the CUDA Runtime API
import numpy as np # Import NumPy for creating data arrays

@cuda.autojit
def hello(ary):
    ary[cuda.threadIdx.x] = cuda.threadIdx.x + cuda.blockIdx.x
    
def main():
    threads_per_block = 1
    number_of_blocks = 1
    ary = np.empty(threads_per_block) # Create an array of threads_per_block elements
    hello[number_of_blocks,threads_per_block] (ary)
    
    print ary # Print out the values filled in by the GPU
    
main()
