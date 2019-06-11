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
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#ifndef N
#define N 4096
#endif

#define sqrt_of_array_cell(x,j) ((half)sqrt(x[j]))

#define FLOAT_N  (3214212.01f)
#define EPS 0.005f

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 256
#define DIM_THREAD_BLOCK_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 32
#define DIM_THREAD_BLOCK_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_THREAD_BLOCK_KERNEL_4_X 256
#define DIM_THREAD_BLOCK_KERNEL_4_Y 1

__global__ void mean_kernel(half2 *mean, half2 *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < N/2)
	{
		mean[j] = __floats2half2_rn(0.0f, 0.0f);

		int i;
		for(i=0; i < N; i++)
		{
			mean[j] = __hadd2(mean[j], data[i*N + j]);
		}
		
		mean[j] = __hmul2(mean[j], __floats2half2_rn(1.0f/FLOAT_N, 1.0f/FLOAT_N));
	}
}


__global__ void std_kernel(half2 *mean, half2 *std, half2 *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < N/2)
	{
		std[j] = __floats2half2_rn(0.0f, 0.0f);

		int i;
		for(i = 0; i < N; i++)
		{
			half2 val = __hsub2(data[i*N + j], mean[j]);

			std[j] = __hfma2(val, val, std[j]);
		}
		std[j] = __hmul2(std[j],  __floats2half2_rn(1.0f/FLOAT_N, 1.0f/FLOAT_N));
		std[j] = h2sqrt(std[j]);
		if(__hle(__low2half(std[j]), __float2half_rz(EPS)))
		{
			std[j] = __halves2half2(__float2half_rz(1.0f), __high2half(std[j]));
		}
		if(__hle(__high2half(std[j]), __float2half_rz(EPS)))
		{
			std[j] = __halves2half2(__low2half(std[j]), __float2half_rz(1.0f));
		}		
	}
}


__global__ void reduce_kernel(half2 *mean, half2 *std, half2 *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < N) && (j < N/2))
	{
		data[i*N + j] = __hsub2(data[i*N + j], mean[j]);
		data[i*N + j] = __hmul2(data[i*N + j], h2rcp(h2sqrt(__hmul2(__floats2half2_rn(FLOAT_N, FLOAT_N), std[j]))));
	}
}


__global__ void corr_kernel(half *symmat, half *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;

	int i, j2;

	if (j1 < (N-1))
	{
		symmat[j1*N + j1] = __float2half_rz(1.0f);

		for (j2 = (j1 + 1); j2 < N; j2++)
		{
			symmat[j1*N + j2] = __float2half_rz(0.0f);

			for(i = 0; i < N; i++)
			{
				symmat[j1*N + j2] = __hfma(data[i*N + j1], data[i*N + j2], symmat[j1*N + j2]);
			}
			symmat[j2*N + j1] = symmat[j1*N + j2];
		}
	}
}


int main()
{
	int i;

	half * data = (half *) malloc(N*N*sizeof(half));
	half * symmat = (half *) malloc(N*N*sizeof(half));	
	half * mean = (half *) malloc(N*sizeof(half));	
	half * stddev = (half *) malloc(N*sizeof(half));	

	srand(5497);
    for (i = 0; i < N*N; i++)
        data[i] = approx_float_to_half((float)rand() / (float)RAND_MAX);


	half *data_gpu;
	half *stddev_gpu;
	half *mean_gpu;
	half *symmat_gpu;

	cudaMalloc((void **)&data_gpu, sizeof(half) * N * N);
	cudaMalloc((void **)&symmat_gpu, sizeof(half) * N * N);
	cudaMalloc((void **)&stddev_gpu, sizeof(half) * N);
	cudaMalloc((void **)&mean_gpu, sizeof(half) * N);
	cudaMemcpy(data_gpu, data, sizeof(half) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(half) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(stddev_gpu, stddev, sizeof(half) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(half) * N, cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)));
	
	dim3 block4(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
	dim3 grid4((size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)), 1);

	mean_kernel<<< grid1, block1 >>>((half2*)mean_gpu,(half2*)data_gpu);
	cudaThreadSynchronize();
	std_kernel<<< grid2, block2 >>>((half2*)mean_gpu,(half2*)stddev_gpu,(half2*)data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<< grid3, block3 >>>((half2*)mean_gpu,(half2*)stddev_gpu,(half2*)data_gpu);
	cudaThreadSynchronize();
	corr_kernel<<< grid4, block4 >>>(symmat_gpu,data_gpu);
	cudaThreadSynchronize();
	

	cudaMemcpy(symmat, symmat_gpu, sizeof(half) * N * N, cudaMemcpyDeviceToHost);

	for (i = 0; i < N*N; i++)
		printf("%.15f,", half_to_float(symmat[i]));


	return 0;
}
