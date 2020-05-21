#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.3989422804

__device__ double cndGPU(double d)
{
    double
        K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
        cnd = RSQRT2PI * exp(- 0.5 * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ void BlackScholesBodyGPU
(
    double& CallResult,
    double& PutResult,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
)
{
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;

    sqrtT = sqrt(T);
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;


    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);
    //printf("%.15f,", CNDD1);

    //Calculate Call and Put simultaneously
    expRT = exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    double *d_CallResult,
    double *d_PutResult,
    double *d_StockPrice,
    double *d_OptionStrike,
    double *d_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
)
{
    //Thread index
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    //Total number of threads in execution grid
    const int THREAD_N = blockDim.x * gridDim.x;

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    for(int opt = tid; opt < optN; opt += THREAD_N)
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


double RandDouble(double low, double high){
    double t = (double)rand() / (double)RAND_MAX;
    return (1.0 - t) * low + t * high;
}

#define OPT_N  400000


const int  NUM_ITERATIONS = 512;


const int          OPT_SZ = OPT_N * sizeof(double);
const double      RISKFREE = 0.02;
const double    VOLATILITY = 0.30;


int main()
{
    double * h_CallResultGPU = (double *)malloc(OPT_SZ);
    double * h_PutResultGPU  = (double *)malloc(OPT_SZ);
    double * h_StockPrice    = (double *)malloc(OPT_SZ);
    double * h_OptionStrike  = (double *)malloc(OPT_SZ);
    double * h_OptionYears   = (double *)malloc(OPT_SZ);


    double
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
        h_CallResultGPU[i] = 0.0;
        h_PutResultGPU[i]  = -1.0;
        h_StockPrice[i]    = RandDouble(5.0, 30.0);
        h_OptionStrike[i]  = RandDouble(1.0, 100.0);
        h_OptionYears[i]   = RandDouble(0.25, 10.0);
    }

    cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice);  

    for(i = 0; i < NUM_ITERATIONS; i++){
        BlackScholesGPU<<<256, 128>>>(
            d_CallResult,
            d_PutResult,
            d_OptionStrike,
            d_StockPrice,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );      
    }

    cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost);   

    //for(i = 0; i < OPT_N; i++)
    //    printf("%.15f,", h_CallResultGPU[i]);

  return 0;
}