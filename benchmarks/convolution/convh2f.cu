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
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#define     KERNEL_RADIUS 8
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
#define      DATA_W  4096
#define      DATA_H  4096
const int   DATA_SIZE = DATA_W * DATA_H * sizeof(half);
const int KERNEL_SIZE = KERNEL_W * sizeof(half);

__device__ __constant__ half d_Kernel[KERNEL_W];

__global__
void convolutionRowGPU(
    half2 *d_Result,
    half2 *d_Data,
    int dataW,
    int dataH,
    int kernelR
){
    int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id
    if(index >= dataW*dataH) return;
    int y = index/dataW;
    int x = index - y*dataW;
    //printf(">>> %d,%d\n", x, y);
    int k, d;
    float2 sum;
    half2 val;


    sum = make_float2(0.0f, 0.0f);
    for(k = -kernelR; k <= kernelR; k++){
        d = x + k;
        if(d >= 0 && d < 2*dataW-1)
        {
            val = __hmul2(d_Data[y * dataW + d], __half2half2(d_Kernel[kernelR - k]));
        }
        else if (d ==- 1)
        {
            val = __floats2half2_rn(0.0f, __high2float(d_Data[y * dataW]) * __half2float(d_Kernel[kernelR - k]));
        }
        else if (d == 2*dataW-1)      
        {
            val = __floats2half2_rn(__low2float(d_Data[y * dataW + d - 1]) * __half2float(d_Kernel[kernelR - k]), 0.0f);        
        }
        sum.x += __low2float(val);
        sum.y += __high2float(val);
    }
    d_Result[y * dataW + x] = __float22half2_rn(sum);
        
}


__global__
void convolutionColumnGPU(
    half2 *d_Result,
    half2 *d_Data,
    int dataW,
    int dataH,
    int kernelR
){
    int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id
    if(index >= dataW*dataH) return;
    int y = index/dataW;
    int x = index - y*dataW;
    int k, d;
    float2 sum;
    half2 val;


    sum = make_float2(0.0f, 0.0f);
    for(k = -kernelR; k <= kernelR; k++){
        d = y + k;
        if(d >= 0 && d < dataH-1)
            val = __hmul2(d_Data[d * dataW + x], __half2half2(d_Kernel[kernelR - k]));
        else if (d ==- 1)
            val = __floats2half2_rn(0.0f, __high2float(d_Data[d * dataW + x]) * __half2float(d_Kernel[kernelR - k]));
        else if (d == dataH-1)      
            val = __floats2half2_rn(__low2float(d_Data[d * dataW + x]) * __half2float(d_Kernel[kernelR - k]), 0.0f);
        sum.x += __low2float(val);
        sum.y += __high2float(val);
    }
    d_Result[y * dataW + x] = __float22half2_rn(sum);
        
}



int main(int argc, char **argv){
    int i;
    
    half
        *h_Kernel,
        *h_DataA;

    half
        *d_DataA,
        *d_DataB;
       

    h_Kernel    = (half *)malloc(KERNEL_SIZE);
    h_DataA     = (half *)malloc(DATA_SIZE);  

    cudaMalloc( (void **)&d_DataA, DATA_SIZE);
    cudaMalloc( (void **)&d_DataB, DATA_SIZE);   

    float kernelSum = 0;
    for(i = 0; i < KERNEL_W; i++){
        float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
        float val = expf(- dist * dist / 2);
        h_Kernel[i] = approx_float_to_half(val);
        kernelSum += val;
    }
    for(i = 0; i < KERNEL_W; i++)
        h_Kernel[i] = approx_float_to_half(half_to_float(h_Kernel[i])/kernelSum);

    srand(5497);
    for(i = 0; i < DATA_W * DATA_H; i++)
        h_DataA[i] = approx_float_to_half((float)rand() / (float)RAND_MAX);

    cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE);
    cudaMemcpy(d_DataA, h_DataA, DATA_SIZE, cudaMemcpyHostToDevice);  

      int blockSize=256;
      int numBlocks = ((DATA_W * DATA_H)/2+blockSize-1)/blockSize;    
      //for(i = 0; i < DATA_W * DATA_H; i++)
       // printf("%.15f,", h_DataA[i]);

    convolutionRowGPU<<<numBlocks, blockSize>>>(
        (half2*)d_DataB,
        (half2*)d_DataA,
        DATA_W/2,
        DATA_H,
        KERNEL_RADIUS
    );
    //cudaMemcpy(h_DataA, d_DataB, DATA_SIZE, cudaMemcpyDeviceToHost);

    //for(i = 0; i < DATA_W * DATA_H; i++)
    //    printf("%.15f,", half_to_float(h_DataA[i]));
    convolutionColumnGPU<<<numBlocks, blockSize>>>(
        (half2*)d_DataA,
        (half2*)d_DataB,
        DATA_W/2,
        DATA_H,
        KERNEL_RADIUS
    );

    cudaMemcpy(h_DataA, d_DataA, DATA_SIZE, cudaMemcpyDeviceToHost);

    for(i = 0; i < DATA_W * DATA_H; i++)
        printf("%.15f,", half_to_float(h_DataA[i]));
    return 0;
}
