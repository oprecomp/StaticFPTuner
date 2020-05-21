#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#define A1 0.31938153f
#define A2 -0.356563782f
#define A3 1.781477937f
#define A4 -1.821255978f
#define A5 1.330274429f
#define RSQRT2PI 0.3989422804f


__device__ half2 absh2(half2 val)
{
  unsigned int val_s = *((unsigned int *)&val);
  val_s = val_s & 0x7FFF7FFF;
  return *((half2 *)&val_s);
}

__device__ float2 cndGPU(float2 d)
{

    float2
        K = make_float2(1.0f / (1.0f + 0.2316419f * fabsf(d.x)),
                        1.0f / (1.0f + 0.2316419f * fabsf(d.y)));


    float2
        cnd = make_float2(RSQRT2PI * expf(- 0.5f * d.x * d.x) * 
        (K.x * (A1 + K.x * (A2 + K.x * (A3 + K.x * (A4 + K.x * A5))))),
        RSQRT2PI * expf(- 0.5f * d.y * d.y) * 
        (K.y * (A1 + K.y * (A2 + K.y * (A3 + K.y * (A4 + K.y * A5))))));


    if(d.x > 0)
        cnd.x = 1.0f - cnd.x;
    if(d.y > 0)
        cnd.y = 1.0f - cnd.y;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ void BlackScholesBodyGPU
(
    float2& CallResult,
    float2& PutResult,
    half2 S, //Stock price
    half2 X, //Option strike
    half2 T, //Option years
    float _R, //Riskless rate
    float _V  //Volatility rate
)
{
    half2 R = __float2half2_rn(_R);
    half2 V = __float2half2_rn(_V);

    half2 sqrtT, expRT;
    half2 d1, d2;
    float2 CNDD1, CNDD2;

    half2 c1 = __float2half2_rn(1.0f);
    half2 c2 = __float2half2_rn(0.5f);

    sqrtT = h2sqrt(T);
    d1 =  __hmul2(__hfma2(T, __hfma2(c2, __hmul2(V, V), R), h2log(__hmul2(S, h2rcp(X)))), h2rcp(__hmul2(V, sqrtT)));
    d2 = __hsub2(d1, __hmul2(V, sqrtT));
    

    CNDD1 = cndGPU(__half22float2(d1));
    CNDD2 = cndGPU(__half22float2(d2));
    //printf("%.15f,%.15f,", __half22float2(CNDD1).x, __half22float2(CNDD1).y);

    //Calculate Call and Put simultaneously
    expRT = h2exp(__hmul2(__hneg2(R), T));
    //float2 temp1 = __half22float2(__hmul2(S, CNDD1));
    float2 temp1 = make_float2(__low2float(S)*CNDD1.x,  __high2float(S)*CNDD1.y);
    //float2 temp2 = __half22float2(__hmul2(X,  __hmul2(expRT, CNDD2)));
    float2 temp2 = make_float2(__low2float(X) * __low2float(expRT) * CNDD2.x, __high2float(X) * __high2float(expRT) * CNDD2.y);
    //float2 temp3 = __half22float2(__hmul2(X, __hmul2(expRT , __hsub2(c1, CNDD2))));
    float2 temp3 = make_float2(__low2float(X) * __low2float(expRT) * (1.0f - CNDD2.x), __high2float(X) * __high2float(expRT) * (1.0f - CNDD2.y));
    //float2 temp4 = __half22float2(__hmul2(S, __hsub2(c1, CNDD1)));
    float2 temp4 = make_float2(__low2float(S) * (1.0f - CNDD1.x), __high2float(S) * (1.0f - CNDD1.y));
    

    CallResult = make_float2(temp1.x-temp2.x, temp1.y-temp2.y);
    PutResult  = make_float2(temp3.x-temp4.x, temp3.y-temp4.y);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float2 *d_CallResult,
    float2 *d_PutResult,
    half2 *d_StockPrice,
    half2 *d_OptionStrike,
    half2 *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    //Thread index
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Total number of threads in execution grid
    const int THREAD_N = blockDim.x * gridDim.x;

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    for(int opt = tid; opt < optN/2; opt += THREAD_N)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}


float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

#define OPT_N  400000


const int  NUM_ITERATIONS = 512;


const int          OPT_SZ  = OPT_N * sizeof(half);
const int          OPT_SZ2 = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;


int main()
{
    float * h_CallResultGPU = (float *)malloc(OPT_SZ2);
    float * h_PutResultGPU  = (float *)malloc(OPT_SZ2);
    half * h_StockPrice    = (half *)malloc(OPT_SZ);
    half * h_OptionStrike  = (half *)malloc(OPT_SZ);
    half * h_OptionYears   = (half *)malloc(OPT_SZ);


    float
        //Results calculated by GPU
        *d_CallResult,
        *d_PutResult;
    half
        //GPU instance of input data
        *d_StockPrice,
        *d_OptionStrike,
        *d_OptionYears;

    cudaMalloc((void **)&d_CallResult,   OPT_SZ2);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ2);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);   

    srand(5347);

    //Generate options set
    int i;
    for(i = 0; i < OPT_N; i++)
    {
        h_CallResultGPU[i] = 0.0f;
        h_PutResultGPU[i]  = -1.0f;
        h_StockPrice[i]    = approx_float_to_half(RandFloat(5.0f, 30.0f));
        h_OptionStrike[i]  = approx_float_to_half(RandFloat(1.0f, 100.0f));
        h_OptionYears[i]   = approx_float_to_half(RandFloat(0.25f, 10.0f));
    }

    cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice);  

    for(i = 0; i < NUM_ITERATIONS; i++){
        BlackScholesGPU<<<256, 128>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (half2 *)d_OptionStrike,
            (half2 *)d_StockPrice,
            (half2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );      
    }

    cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ2, cudaMemcpyDeviceToHost);   

    //for(i = 0; i < OPT_N; i++)
    //    printf("%.15f,", h_CallResultGPU[i]);

  return 0;
}