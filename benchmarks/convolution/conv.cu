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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef FLOAT
#define FLOAT double
#endif

#define     KERNEL_RADIUS 8
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
#define      DATA_W  4096
#define      DATA_H  4096
const int   DATA_SIZE = DATA_W * DATA_H * sizeof(FLOAT);
const int KERNEL_SIZE = KERNEL_W * sizeof(FLOAT);

__device__ __constant__ FLOAT d_Kernel[KERNEL_W];

__global__
void convolutionRowGPU(
    FLOAT *d_Result,
    FLOAT *d_Data,
    int dataW,
    int dataH,
    int kernelR
){
    int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id
    if(index >= dataW*dataH) return;
    int y = index/dataW;
    int x = index - y*dataW;
    int k, d;
    FLOAT sum;


    sum = FLOAT(0.0f);
    for(k = -kernelR; k <= kernelR; k++){
        d = x + k;
        if(d >= 0 && d < dataW)
            sum += d_Data[y * dataW + d] * d_Kernel[kernelR - k];
    }
    d_Result[y * dataW + x] = sum;
        
}


__global__
void convolutionColumnGPU(
    FLOAT *d_Result,
    FLOAT *d_Data,
    int dataW,
    int dataH,
    int kernelR
){
    int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id
    if(index >= dataW*dataH) return;
    int y = index/dataW;
    int x = index - y*dataW;
    int k, d;
    FLOAT sum;


    sum = FLOAT(0.0f);
    for(k = -kernelR; k <= kernelR; k++){
        d = y + k;
        if(d >= 0 && d < dataH)
            sum += d_Data[d * dataW + x] * d_Kernel[kernelR - k];
    }
    d_Result[y * dataW + x] = sum;
        
}



int main(int argc, char **argv){
    int i;
    
    FLOAT
        *h_Kernel,
        *h_DataA;

    FLOAT
        *d_DataA,
        *d_DataB;
       

    h_Kernel    = (FLOAT *)malloc(KERNEL_SIZE);
    h_DataA     = (FLOAT *)malloc(DATA_SIZE);  

    cudaMalloc( (void **)&d_DataA, DATA_SIZE);
    cudaMalloc( (void **)&d_DataB, DATA_SIZE);   

    FLOAT kernelSum = 0;
    for(i = 0; i < KERNEL_W; i++){
        FLOAT dist = (FLOAT)(i - KERNEL_RADIUS) / (FLOAT)KERNEL_RADIUS;
        h_Kernel[i] = expf(- dist * dist / 2);
        kernelSum += h_Kernel[i];
    }
    for(i = 0; i < KERNEL_W; i++)
        h_Kernel[i] /= kernelSum;

    srand(5497);
    for(i = 0; i < DATA_W * DATA_H; i++)
        h_DataA[i] = (FLOAT)rand() / (FLOAT)RAND_MAX;

    cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE);
    cudaMemcpy(d_DataA, h_DataA, DATA_SIZE, cudaMemcpyHostToDevice);  

  int blockSize=256;
  int numBlocks = ((DATA_W * DATA_H)+blockSize-1)/blockSize;    
      //for(i = 0; i < DATA_W * DATA_H; i++)
       // printf("%.15f,", h_DataA[i]);

    convolutionRowGPU<<<numBlocks, blockSize>>>(
        d_DataB,
        d_DataA,
        DATA_W,
        DATA_H,
        KERNEL_RADIUS
    );
    //cudaMemcpy(h_DataA, d_DataB, DATA_SIZE, cudaMemcpyDeviceToHost);

    //for(i = 0; i < DATA_W * DATA_H; i++)
    //    printf("%.15f,", h_DataA[i]);
    convolutionColumnGPU<<<numBlocks, blockSize>>>(
        d_DataA,
        d_DataB,
        DATA_W,
        DATA_H,
        KERNEL_RADIUS
    );

    cudaMemcpy(h_DataA, d_DataA, DATA_SIZE, cudaMemcpyDeviceToHost);

    for(i = 0; i < DATA_W * DATA_H; i++)
        printf("%.15f,", h_DataA[i]);
    return 0;
}
