/*
    Copyright 2018 - The OPRECOMP Project Consortium, Alma Mater Studiorum
    Università di Bologna. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef N
#define N 256
#endif

#ifndef FLOAT
#define FLOAT double
#endif

#define sqrt_of_array_cell(x,j) ((FLOAT)sqrt(x[j]))

#define FLOAT_N 3214212.01f
#define EPS 0.005f

#include "datasets.h"

void correlation(FLOAT *data, FLOAT *mean, FLOAT *stddev, FLOAT *symmat)
{
	int i, j, j1, j2;

	// Determine mean of column vectors of input data matrix
  	for (j = 0; j < N; j++)
   	{
  		mean[j] = 0.0f;

   		for (i = 0; i < N; i++)
		{
			mean[j] += data[i*N+j];
   		}

		mean[j] /= (FLOAT)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix.
  	for (j = 0; j < N; j++)
   	{
   		stddev[j] = 0.0f;

		for (i = 0; i < N; i++)
		{
			stddev[j] += (data[i*N+j] - mean[j]) * (data[i*N+j] - mean[j]);
		}

		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0f : stddev[j];
	}

 	// Center and reduce the column vectors.
  	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			data[i*N+j] -= mean[j];
			data[i*N+j] /= (sqrt(FLOAT_N)*stddev[j]) ;
		}
	}

	// Calculate the m * m correlation matrix.
  	for (j1 = 0; j1 < N-1; j1++)
	{
		symmat[j1*N+j1] = 1.0f;

		for (j2 = j1+1; j2 < N; j2++)
		{
	  		symmat[j1*N+j2] = 0.0f;

	  		for (i = 0; i < N; i++)
			{
	   			symmat[j1*N+j2] += (data[i*N+j1] * data[i*N+j2]);
			}

	  		symmat[j2*N+j1] = symmat[j1*N+j2];
		}
	}

	symmat[(N-1)*N + (N-1)] = 1.0f;
}

int main()
{
	int i;

	FLOAT * symmat = (FLOAT *) malloc(N*N*sizeof(FLOAT));
	FLOAT * mean = (FLOAT *) malloc(N*sizeof(FLOAT));
	FLOAT * stddev = (FLOAT *) malloc(N*sizeof(FLOAT));


  correlation(data, mean, stddev, symmat);

	for (i = 0; i < N*N; i++)
		printf("%.15f,", symmat[i]);


	return 0;
}
