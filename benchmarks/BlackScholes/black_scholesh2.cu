#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#define A1 (__float2half2_rn(0.31938153))
#define A2 (__float2half2_rn(-0.356563782))
#define A3 (__float2half2_rn(1.781477937))
#define A4 (__float2half2_rn(-1.821255978))
#define A5 (__float2half2_rn(1.330274429))
#define RSQRT2PI (__float2half2_rn(0.3989422804))


__device__ half2 absh2(half2 val)
{
  unsigned int val_s = *((unsigned int *)&val);
  val_s = val_s & 0x7FFF7FFF;
  return *((half2 *)&val_s);
}

__device__ half2 cndGPU(half2 d)
{
    half2 c1 = __float2half2_rn(1.0f);
    half2 c2 = __float2half2_rn(0.2316419f);
    half2 c3 = __float2half2_rn(-0.5f);
    half2 c4 = __float2half2_rn(0.0);

    half2
        K = h2rcp(__hfma2(c2, absh2(d), c1));


    half2
        cnd = __hmul2(RSQRT2PI, __hmul2(h2exp(__hmul2(c3, __hmul2(d, d))),
        (__hmul2(K, __hfma2(K,__hfma2(K, __hfma2(K, __hfma2(K, A5, A4), A3), A2), A1)))));



//    if(d > 0)
//        cnd = 1.0 - cnd;
    half2 val_gt = __hgt2(d, c4);
    //unsigned int temp = ((*((unsigned int *)&val_gt) & 0x2FFF2FFF) << 2) ^ *((unsigned int *)&cnd);
    //cnd = *((half2 *)&temp);
    //cnd = __hadd2(val_gt, cnd);
    half one = __low2half(c1);
    half cnd_l = __low2half(cnd);
    half cnd_h = __high2half(cnd);
    if(__heq(__low2half(val_gt), one))
    {
        cnd_l = __hsub(one,cnd_l);
    }
    if(__heq(__high2half(val_gt), one))
    {
        cnd_h = __hsub(one,cnd_h);
    }    

    return __halves2half2(cnd_l, cnd_h);
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ void BlackScholesBodyGPU
(
    half2& CallResult,
    half2& PutResult,
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
    half2 d1, d2, CNDD1, CNDD2;

    half2 c1 = __float2half2_rn(1.0f);
    half2 c2 = __float2half2_rn(0.5f);

    sqrtT = h2sqrt(T);
    d1 =  __hmul2(__hfma2(T, __hfma2(c2, __hmul2(V, V), R), h2log(__hmul2(S, h2rcp(X)))), h2rcp(__hmul2(V, sqrtT)));
    d2 = __hsub2(d1, __hmul2(V, sqrtT));
    

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);
    //printf("%.15f,%.15f,", __half22float2(CNDD1).x, __half22float2(CNDD1).y);

    //Calculate Call and Put simultaneously
    expRT = h2exp(__hmul2(__hneg2(R), T));
    CallResult = __hsub2(__hmul2(S, CNDD1), __hmul2(X,  __hmul2(expRT, CNDD2)));
    PutResult  = __hsub2(__hmul2(X, __hmul2(expRT , __hsub2(c1, CNDD2))), __hmul2(S, __hsub2(c1, CNDD1)));
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    half2 *d_CallResult,
    half2 *d_PutResult,
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


const int          OPT_SZ = OPT_N * sizeof(half);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;


int main()
{
    half * h_CallResultGPU = (half *)malloc(OPT_SZ);
    half * h_PutResultGPU  = (half *)malloc(OPT_SZ);
    half * h_StockPrice    = (half *)malloc(OPT_SZ);
    half * h_OptionStrike  = (half *)malloc(OPT_SZ);
    half * h_OptionYears   = (half *)malloc(OPT_SZ);


    half
        //Results calculated by GPU
        *d_CallResult,
        *d_PutResult,
        //GPU instance of input data
        *d_StockPrice,
        *d_OptionStrike,
        *d_OptionYears;

    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);   

    srand(5347);

    //Generate options set
    int i;
    for(i = 0; i < OPT_N; i++)
    {
        h_CallResultGPU[i] = approx_float_to_half(0.0f);
        h_PutResultGPU[i]  = approx_float_to_half(-1.0f);
        h_StockPrice[i]    = approx_float_to_half(RandFloat(5.0f, 30.0f));
        h_OptionStrike[i]  = approx_float_to_half(RandFloat(1.0f, 100.0f));
        h_OptionYears[i]   = approx_float_to_half(RandFloat(0.25f, 10.0f));
    }

    cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice);  

    for(i = 0; i < NUM_ITERATIONS; i++){
        BlackScholesGPU<<<256, 128>>>(
            (half2 *)d_CallResult,
            (half2 *)d_PutResult,
            (half2 *)d_OptionStrike,
            (half2 *)d_StockPrice,
            (half2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );      
    }

    cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost);   

    //for(i = 0; i < OPT_N; i++)
    //    printf("%.15f,", half_to_float(h_CallResultGPU[i]));

  return 0;
}