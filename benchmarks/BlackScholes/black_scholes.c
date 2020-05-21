#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.3989422804

#include "datasets.h"

double cndCPU(double d)
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
void BlackScholesBodyCPU(
    double* CallResult,
    double* PutResult,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
){
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;

    sqrtT = sqrt(T);
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndCPU(d1);
    CNDD2 = cndCPU(d2);

    //Calculate Call and Put simultaneously
    expRT = exp(- R * T);
    *CallResult = S * CNDD1 - X * expRT * CNDD2;
    *PutResult  = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    double *h_CallResult,
    double *h_PutResult,
    double *h_StockPrice,
    double *h_OptionStrike,
    double *h_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
){
    for(int opt = 0; opt < optN; opt++)
        BlackScholesBodyCPU(
            &h_CallResult[opt],
            &h_PutResult[opt],
            h_StockPrice[opt],
            h_OptionStrike[opt],
            h_OptionYears[opt],
            Riskfree,
            Volatility
        );
}

double RandDouble(double low, double high){
    double t = (double)rand() / (double)RAND_MAX;
    return (1.0 - t) * low + t * high;
}

#define OPT_N  4000

#ifdef __DEVICE_EMULATION__
const int  NUM_ITERATIONS = 1;
#else
const int  NUM_ITERATIONS = 512;
#endif


const int          OPT_SZ = OPT_N * sizeof(double);
const double      RISKFREE = 0.02;
const double    VOLATILITY = 0.30;


int main()
{
    double * h_CallResultCPU = (double *)malloc(OPT_SZ);
    double * h_PutResultCPU  = (double *)malloc(OPT_SZ);
    // double * h_StockPrice    = (double *)malloc(OPT_SZ);
    // double * h_OptionStrike  = (double *)malloc(OPT_SZ);
    // double * h_OptionYears   = (double *)malloc(OPT_SZ);

    // srand(5347);

    //Generate options set
    int i;
    for(i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0;
        h_PutResultCPU[i]  = -1.0;
        // h_StockPrice[i]    = RandDouble(5.0, 30.0);
        // h_OptionStrike[i]  = RandDouble(1.0, 100.0);
        // h_OptionYears[i]   = RandDouble(0.25, 10.0);
    }

/*
    printf("double h_StockPrice[]={");
    for(i = 0; i < OPT_N; i++)
        printf("%.15f,", h_StockPrice[i]);
    printf("};\n");

    printf("double h_OptionStrike[]={");
    for(i = 0; i < OPT_N; i++)
        printf("%.15f,", h_OptionStrike[i]);
    printf("};\n");

    printf("double h_OptionYears[]={");
    for(i = 0; i < OPT_N; i++)
        printf("%.15f,", h_OptionYears[i]);
    printf("};\n");
*/
    for(i = 0; i < NUM_ITERATIONS; i++){
      BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_OptionStrike,
        h_StockPrice,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N);
    }

    for(i = 0; i < OPT_N; i++)
        printf("%.15f,", h_CallResultCPU[i]);

  return 0;
}
