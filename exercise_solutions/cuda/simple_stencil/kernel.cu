/*
 *  Copyright 2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdio.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif

#define N 4000000
#define RADIUS 5
#define THREADS_PER_BLOCK 512

__global__ void stencil_1d(int n, double *in, double *out)
{
/* calculate global index in the array */
  int gindex = blockIdx.x * blockDim.x + threadIdx.x;
	
/* return if my global index is larger than the array size */
  if( gindex >= n ) return;

/* code to handle the boundary conditions */
  if( gindex < RADIUS || gindex >= (n - RADIUS) ) 
  {
    out[gindex] = (double) gindex * ( (double)RADIUS*2 + 1) ;
    return;
  } /* end if */

  double result = 0.0;
	
  for( int i = gindex-(RADIUS); i <= gindex+(RADIUS); i++ ) 
  {
    result += in[i];
  }

  out[gindex] = result;
  return;

}

int main()
{
  double *in, *out;
  double *d_in, *d_out;
  int size = N * sizeof( double );

/* allocate space for device copies of in, out */

  CUDA_CALL( cudaMalloc( (void **) &d_in, size ) );
  CUDA_CALL( cudaMalloc( (void **) &d_out, size ) );

/* allocate space for host copies of in, out and setup input values */

  in = (double *)malloc( size );
  out = (double *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    in[i] = (double) i;
    out[i] = 0;
  }

/* copy inputs to device */

  CUDA_CALL( cudaMemcpy( d_in, in, size, cudaMemcpyHostToDevice ) );
  CUDA_CALL( cudaMemset( d_out, 0, size ) );

/* calculate block and grid sizes */

  dim3 threads( THREADS_PER_BLOCK, 1, 1);
  dim3 blocks( (N / threads.x) + 1, 1, 1);

/* start the timers */

  cudaEvent_t start, stop;
  CUDA_CALL( cudaEventCreate( &start ) );
  CUDA_CALL( cudaEventCreate( &stop ) );
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* launch the kernel on the GPU */

  stencil_1d<<< blocks, threads >>>( N, d_in, d_out );
  CUDA_CHECK();
  CUDA_CALL( cudaDeviceSynchronize() );

/* stop the timers */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  float elapsedTime;
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

  printf("Total time for %d elements was %f ms\n", N, elapsedTime );

/* copy result back to host */

  CUDA_CALL( cudaMemcpy( out, d_out, size, cudaMemcpyDeviceToHost ) );

  for( int i = 0; i < N; i++ )
  {
    if( in[i]*( (double)RADIUS*2+1 ) != out[i] ) 
      printf("error in element %d in = %f out %f\n",i,in[i],out[i] );
  } /* end for */

/* clean up */

  free(in);
  free(out);
  CUDA_CALL( cudaFree( d_in ) );
  CUDA_CALL( cudaFree( d_out ) );

  CUDA_CALL( cudaDeviceReset() );
	
  return 0;
} /* end main */
