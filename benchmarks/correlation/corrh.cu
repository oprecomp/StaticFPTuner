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
#define N 256
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

__global__ void mean_kernel(half *mean, half *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < N)
	{
		mean[j] = __float2half_rz(0.0f);

		int i;
		for(i=0; i < N; i++)
		{
			mean[j] = __hadd(mean[j], data[i*N + j]);
		}
		
		mean[j] = __hmul(mean[j], __float2half_rz(1.0f/FLOAT_N));
	}
}


__global__ void std_kernel(half *mean, half *std, half *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < N)
	{
		std[j] = __float2half_rz(0.0f);

		int i;
		for(i = 0; i < N; i++)
		{
			half val = __hsub(data[i*N + j], mean[j]);

			std[j] = __hfma(val, val, std[j]);
		}
		std[j] = __hmul(std[j],  __float2half_rz(1.0f/FLOAT_N));
		std[j] = hsqrt(std[j]);
		if(__hle(std[j], __float2half_rz(EPS)))
		{
			std[j] = __float2half_rz(1.0f);
		}
	}
}


__global__ void reduce_kernel(half *mean, half *std, half *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < N) && (j < N))
	{
		data[i*N + j] = __hsub(data[i*N + j], mean[j]);
		data[i*N + j] = __hmul(data[i*N + j], hrcp(hsqrt(__hmul(__float2half_rz(FLOAT_N), std[j]))));
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

	mean_kernel<<< grid1, block1 >>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	std_kernel<<< grid2, block2 >>>(mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<< grid3, block3 >>>(mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	corr_kernel<<< grid4, block4 >>>(symmat_gpu,data_gpu);
	cudaThreadSynchronize();
	

	cudaMemcpy(symmat, symmat_gpu, sizeof(half) * N * N, cudaMemcpyDeviceToHost);

	for (i = 0; i < N*N; i++)
		printf("%.15f,", half_to_float(symmat[i]));


	return 0;
}
