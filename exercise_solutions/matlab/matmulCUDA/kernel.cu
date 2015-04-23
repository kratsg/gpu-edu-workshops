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
#include "cublas_v2.h"

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* naive GPU kernel where each element of C is computed by a single thread */

__global__ void GPU_matmul( double * const c, const int m, 
                            double const * const a,  
                            double const * const b )
{

/* determine my threads's row and col indices in the global C matrix */

  const int myrow = blockDim.x * blockIdx.x + threadIdx.x;
  const int mycol = blockDim.y * blockIdx.y + threadIdx.y;

/* if my row and col are in the C matrix, then calculate that value of C */

  if( myrow < m && mycol < m )
  {
    register double temp = 0.0;

    for( int k = 0; k < m; k++ ) 
      temp += a[INDX( myrow, k, m )] * b[INDX( k, mycol, m )];

    c[INDX( myrow, mycol, m )] = temp;
  } /* end if */

	return;
} /* end GPU_naive */

