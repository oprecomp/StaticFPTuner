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

void convolutionRowCPU(
    FLOAT *h_Result,
    FLOAT *h_Data,
    FLOAT *h_Kernel,
    int dataW,
    int dataH,
    int kernelR
){
    int x, y, k, d;
    FLOAT sum;

    for(y = 0; y < dataH; y++)
        for(x = 0; x < dataW; x++){
            sum = 0;
            for(k = -kernelR; k <= kernelR; k++){
                d = x + k;
                if(d >= 0 && d < dataW)
                {
                    //printf("[%d %d %d] %f %f\n", x, y, k, h_Data[y * dataW + d], h_Kernel[kernelR - k]);
                    sum += h_Data[y * dataW + d] * h_Kernel[kernelR - k];
                }
            }
            h_Result[y * dataW + x] = sum;
        }
}



void convolutionColumnCPU(
    FLOAT *h_Result,
    FLOAT *h_Data,
    FLOAT *h_Kernel,
    int dataW,
    int dataH,
    int kernelR
){
    int x, y, k, d;
    FLOAT sum;

    for(y = 0; y < dataH; y++)
        for(x = 0; x < dataW; x++){
            sum = 0;
            for(k = -kernelR; k <= kernelR; k++){
                d = y + k;
                if(d >= 0 && d < dataH)
                    sum += h_Data[d * dataW + x] * h_Kernel[kernelR - k];
            }
            h_Result[y * dataW + x] = sum;
        }
}


#define     KERNEL_RADIUS 8
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
#define      DATA_W  256
#define      DATA_H  256
const int   DATA_SIZE = DATA_W * DATA_H * sizeof(FLOAT);
const int KERNEL_SIZE = KERNEL_W * sizeof(FLOAT);

#include "datasets.h"

int main(int argc, char **argv){
    int i;

    FLOAT
        *h_Kernel,
        *h_DataB,
        *h_ResultGPU;

    h_Kernel    = (FLOAT *)malloc(KERNEL_SIZE);
    h_DataB     = (FLOAT *)malloc(DATA_SIZE);

    FLOAT kernelSum = 0;
    for(i = 0; i < KERNEL_W; i++){
        FLOAT dist = (FLOAT)(i - KERNEL_RADIUS) / (FLOAT)KERNEL_RADIUS;
        h_Kernel[i] = expf(- dist * dist / 2);
        kernelSum += h_Kernel[i];
    }
    for(i = 0; i < KERNEL_W; i++)
        h_Kernel[i] /= kernelSum;

    convolutionRowCPU(
        h_DataB,
        data,
        h_Kernel,
        DATA_W,
        DATA_H,
        KERNEL_RADIUS
    );

    convolutionColumnCPU(
        data,
        h_DataB,
        h_Kernel,
        DATA_W,
        DATA_H,
        KERNEL_RADIUS
    );
    for(i = 0; i < DATA_W * DATA_H; i++)
        printf("%.15f,", data[i]);
    return 0;
}
