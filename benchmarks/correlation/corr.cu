#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifndef N
#define N 4096
#endif

#ifndef FLOAT
#define FLOAT double
#endif

#define sqrt_of_array_cell(x,j) ((FLOAT)sqrt(x[j]))

#define FLOAT_N 3214212.01f
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

__global__ void mean_kernel(FLOAT *mean, FLOAT *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < N)
	{
		mean[j] = 0.0f;

		int i;
		for(i=0; i < N; i++)
		{
			mean[j] += data[i*N + j];
		}
		
		mean[j] /= (FLOAT)FLOAT_N;
	}
}


__global__ void std_kernel(FLOAT *mean, FLOAT *std, FLOAT *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < N)
	{
		std[j] = 0.0f;

		int i;
		for(i = 0; i < N; i++)
		{
			std[j] += (data[i*N + j] - mean[j]) * (data[i*N + j] - mean[j]);
		}
		std[j] /= (FLOAT_N);
		std[j] = FLOAT(sqrt(std[j]));
		if(std[j] <= EPS) 
		{
			std[j] = 1.0f;
		}
	}
}


__global__ void reduce_kernel(FLOAT *mean, FLOAT *std, FLOAT *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < N) && (j < N))
	{
		data[i*N + j] -= mean[j];
		data[i*N + j] /= FLOAT(sqrt(FLOAT_N)) * std[j];
	}
}


__global__ void corr_kernel(FLOAT *symmat, FLOAT *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;

	int i, j2;

	if (j1 < (N-1))
	{
		symmat[j1*N + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < N; j2++)
		{
			symmat[j1*N + j2] = 0.0;

			for(i = 0; i < N; i++)
			{
				symmat[j1*N + j2] += data[i*N + j1] * data[i*N + j2];
			}
			symmat[j2*N + j1] = symmat[j1*N + j2];
		}
	}
}


int main()
{
	int i;

	FLOAT * data = (FLOAT *) malloc(N*N*sizeof(FLOAT));
	FLOAT * symmat = (FLOAT *) malloc(N*N*sizeof(FLOAT));	
	FLOAT * mean = (FLOAT *) malloc(N*sizeof(FLOAT));	
	FLOAT * stddev = (FLOAT *) malloc(N*sizeof(FLOAT));	

	srand(5497);
    for (i = 0; i < N*N; i++)
        data[i] = (FLOAT)rand() / (FLOAT)RAND_MAX;


	FLOAT *data_gpu;
	FLOAT *stddev_gpu;
	FLOAT *mean_gpu;
	FLOAT *symmat_gpu;

	cudaMalloc((void **)&data_gpu, sizeof(FLOAT) * N * N);
	cudaMalloc((void **)&symmat_gpu, sizeof(FLOAT) * N * N);
	cudaMalloc((void **)&stddev_gpu, sizeof(FLOAT) * N);
	cudaMalloc((void **)&mean_gpu, sizeof(FLOAT) * N);
	cudaMemcpy(data_gpu, data, sizeof(FLOAT) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(FLOAT) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(stddev_gpu, stddev, sizeof(FLOAT) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(FLOAT) * N, cudaMemcpyHostToDevice);

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
	

	cudaMemcpy(symmat, symmat_gpu, sizeof(FLOAT) * N * N, cudaMemcpyDeviceToHost);

	for (i = 0; i < N*N; i++)
		printf("%.15f,", symmat[i]);


	return 0;
}
