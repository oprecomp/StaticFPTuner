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
#include "flexfloat.hpp"

#ifndef N
#define N 256
#endif

#ifndef FLOAT
#define FLOAT double
#endif

#define sqrt_of_array_cell(x,j) ((FLOAT)sqrt(x[j]))
#define SQRT(x) (sqrt(x))

#define FLOAT_N 3214212.01f
#define EPS 0.005f

#include "datasets.h"

void correlation(FLOAT *data, FLOAT *mean, FLOAT *stddev, FLOAT *symmat)
{
	int i, j, j1, j2;

  flexfloat<EXP_DATA, FRAC_DATA> ff_data, ff_data2;
  flexfloat<EXP_MEAN, FRAC_MEAN> ff_mean;
  flexfloat<EXP_STDDEV, FRAC_STDDEV> ff_stddev;
  flexfloat<EXP_SYMMAT, FRAC_SYMMAT> ff_symmat;

  // Determine mean of column vectors of input data matrix
  for (j = 0; j < N; j++)
  {
     ff_mean = 0.0f;

    for (i = 0; i < N; i++)
    {
      ff_data = data[i*N+j]; // [/SKIP]
      ff_mean += flexfloat<EXP_MEAN, FRAC_MEAN>(ff_data);
    }

     ff_mean /= (FLOAT)FLOAT_N;
     mean[j] = FLOAT(ff_mean); // [/SKIP]
   }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 0; j < N; j++)
  {
    ff_stddev = 0.0f;

    for (i = 0; i < N; i++)
    {
      ff_mean = mean[j]; // [/SKIP]
      ff_data = data[i*N+j]; // [/SKIP]
      ff_stddev += flexfloat<EXP_STDDEV, FRAC_STDDEV>(flexfloat<EXP_TEMP1, FRAC_TEMP1>(ff_data) - flexfloat<EXP_TEMP1, FRAC_TEMP1>(ff_mean)) *
                   flexfloat<EXP_STDDEV, FRAC_STDDEV>(flexfloat<EXP_TEMP1, FRAC_TEMP1>(ff_data) - flexfloat<EXP_TEMP1, FRAC_TEMP1>(ff_mean)) ;
    }

    ff_stddev /= FLOAT_N;
    ff_stddev = flexfloat<EXP_STDDEV, FRAC_STDDEV>(flexfloat<EXP_TEMP2, FRAC_TEMP2>(SQRT(FLOAT(ff_stddev))));
    stddev[j] = FLOAT(ff_stddev); // [/SKIP]
    stddev[j] = stddev[j] <= EPS ? 1.0f : stddev[j];
  }

  // Center and reduce the column vectors.
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      ff_data = data[i*N+j]; // [/SKIP]
      ff_mean = mean[j]; // [/SKIP]
      ff_stddev = stddev[j]; // [/SKIP]
      ff_data -= flexfloat<EXP_DATA, FRAC_DATA>(ff_mean);
      ff_data /= flexfloat<EXP_DATA, FRAC_DATA>((flexfloat<EXP_TEMP3, FRAC_TEMP3>(SQRT(FLOAT_N))*flexfloat<EXP_TEMP3, FRAC_TEMP3>(ff_stddev)));
      data[i*N+j] = FLOAT(ff_data); // [/SKIP]
    }
  }

  // Calculate the m * m correlation matrix.
  for (j1 = 0; j1 < N-1; j1++)
  {
    symmat[j1*N+j1] = 1.0f;

    for (j2 = j1+1; j2 < N; j2++)
    {
      ff_symmat = 0.0f;

      for (i = 0; i < N; i++)
      {
        ff_data  = data[i*N+j1]; // [/SKIP]
        ff_data2 = data[i*N+j2]; // [/SKIP]
        ff_symmat += flexfloat<EXP_SYMMAT, FRAC_SYMMAT>(ff_data*ff_data2);
      }
      symmat[j1*N+j2] = FLOAT(ff_symmat); // [/SKIP]

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
