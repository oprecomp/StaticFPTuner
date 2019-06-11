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
#include <cuda_fp16.h>

#define SIZE 1000000

#define nTPB 256

#ifndef FLOAT
#define FLOAT double
#endif

__global__ void saxpy(int n, FLOAT a, FLOAT *x, FLOAT *y)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < n)
  {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main()
{

  FLOAT *hin, *hout, *din, *dout;
  hin  = (FLOAT *)malloc(SIZE*sizeof(FLOAT));
  hout = (FLOAT *)malloc(SIZE*sizeof(FLOAT));
  for (int i = 0; i < SIZE; i++) hin[i] = i%15;
  for (int i = 0; i < SIZE; i++) hout[i] = i%15;
  cudaMalloc(&din,  SIZE*sizeof(FLOAT));
  cudaMalloc(&dout, SIZE*sizeof(FLOAT));
  cudaMemcpy(din, hin, SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
  cudaMemcpy(dout, hout, SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);

  int k;
  for(k=0; k<5; ++k)
    saxpy<<<(SIZE+nTPB-1)/nTPB,nTPB>>>(SIZE, 0.5124353, din, dout);
  cudaMemcpy(hout, dout, SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);
  for (int i = 0; i < SIZE; i++)
    printf("%f,", hout[i]);
  return 0;
}

