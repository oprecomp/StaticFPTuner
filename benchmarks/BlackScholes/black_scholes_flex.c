#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <flexfloat.hpp>

#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.3989422804

#include "datasets.h"

// Used to enforce variable ordering (for analysis tools)
flexfloat<EXP_K, FRAC_K> _v1;
flexfloat<EXP_CND, FRAC_CND> _v2;
flexfloat<EXP_D, FRAC_D> _v3;
flexfloat<EXP_SQRTT, FRAC_SQRTT> _v4;
flexfloat<EXP_EXPRT, FRAC_EXPRT> _v5;
flexfloat<EXP_SX, FRAC_SX> _v6;
flexfloat<EXP_RV, FRAC_RV> _v7;
flexfloat<EXP_T, FRAC_T> _v8;
flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT> _v9;
flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT> _v10;
flexfloat<EXP_TEMP1, FRAC_TEMP1> _v11;
flexfloat<EXP_TEMP2, FRAC_TEMP2> _v12;
flexfloat<EXP_TEMP3, FRAC_TEMP3> _v13;
flexfloat<EXP_TEMP4, FRAC_TEMP4> _v14;
flexfloat<EXP_TEMP5, FRAC_TEMP5> _v15;


flexfloat<EXP_CND, FRAC_CND> cndCPU(flexfloat<EXP_D, FRAC_D> d)
{
    flexfloat<EXP_K, FRAC_K> K;
    flexfloat<EXP_CND, FRAC_CND> cnd;

    K = flexfloat<EXP_K, FRAC_K>(flexfloat<EXP_TEMP1, FRAC_TEMP1>(1.0) / (flexfloat<EXP_TEMP1, FRAC_TEMP1>(1.0) + flexfloat<EXP_TEMP1, FRAC_TEMP1>(0.2316419) * flexfloat<EXP_TEMP1, FRAC_TEMP1>(fabs(double(d)))));

    cnd = flexfloat<EXP_CND, FRAC_CND>(flexfloat<EXP_TEMP2, FRAC_TEMP2>(RSQRT2PI) * flexfloat<EXP_TEMP2, FRAC_TEMP2>(exp(double(flexfloat<EXP_D, FRAC_D>(- 0.5) * d * d))) *
        flexfloat<EXP_TEMP2, FRAC_TEMP2>((K * (flexfloat<EXP_K, FRAC_K>(A1) + K * (flexfloat<EXP_K, FRAC_K>(A2) + K * (flexfloat<EXP_K, FRAC_K>(A3) + K * (flexfloat<EXP_K, FRAC_K>(A4) + K * flexfloat<EXP_K, FRAC_K>(A5))))))));

    if(d > flexfloat<EXP_D, FRAC_D>(0))
        cnd = flexfloat<EXP_CND, FRAC_CND>(1.0) - cnd;

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
    flexfloat<EXP_SQRTT, FRAC_SQRTT> sqrtT;
    flexfloat<EXP_EXPRT, FRAC_EXPRT> expRT;
    flexfloat<EXP_D, FRAC_D> d1;
    flexfloat<EXP_D, FRAC_D> d2;
    flexfloat<EXP_CND, FRAC_CND> CNDD1;
    flexfloat<EXP_CND, FRAC_CND> CNDD2;
    flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT> ff_CallResult;
    flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT> ff_PutResult;

    sqrtT = flexfloat<EXP_SQRTT, FRAC_SQRTT>(sqrt(double(flexfloat<EXP_T, FRAC_T>(T))));
    d1 = flexfloat<EXP_D, FRAC_D>(
    	   (flexfloat<EXP_TEMP3, FRAC_TEMP3>(log(double(flexfloat<EXP_SX, FRAC_SX>(S) / flexfloat<EXP_SX, FRAC_SX>(X)))) +
    	   	flexfloat<EXP_TEMP3, FRAC_TEMP3>((flexfloat<EXP_RV, FRAC_RV>(R) + flexfloat<EXP_RV, FRAC_RV>(0.5) * flexfloat<EXP_RV, FRAC_RV>(V) * flexfloat<EXP_RV, FRAC_RV>(V))) *
    	   	flexfloat<EXP_TEMP3, FRAC_TEMP3>(flexfloat<EXP_T, FRAC_T>(T)))
    	   /
    	   flexfloat<EXP_TEMP3, FRAC_TEMP3>(flexfloat<EXP_RV, FRAC_RV>(V) * flexfloat<EXP_RV, FRAC_RV>(sqrtT))
    );
    d2 = d1 - flexfloat<EXP_D, FRAC_D>(flexfloat<EXP_TEMP4, FRAC_TEMP4>(V) * flexfloat<EXP_TEMP4, FRAC_TEMP4>(sqrtT));

    CNDD1 = cndCPU(d1);
    CNDD2 = cndCPU(d2);

    //Calculate Call and Put simultaneously
    expRT = flexfloat<EXP_EXPRT, FRAC_EXPRT>(exp(double(flexfloat<EXP_TEMP5, FRAC_TEMP5>(- flexfloat<EXP_RV, FRAC_RV>(R)) * flexfloat<EXP_TEMP5, FRAC_TEMP5>(flexfloat<EXP_T, FRAC_T>(T)))));
    ff_CallResult = flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT>(flexfloat<EXP_SX, FRAC_SX>(S)) * flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT>(CNDD1) - flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT>(X) * flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT>(expRT) * flexfloat<EXP_CALLRESULT, FRAC_CALLRESULT>(CNDD2);
    ff_PutResult  = flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT>(flexfloat<EXP_SX, FRAC_SX>(X)) * flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT>(expRT) * flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT>((flexfloat<EXP_CND, FRAC_CND>(1.0) - CNDD2)) - flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT>(flexfloat<EXP_SX, FRAC_SX>(S)) * flexfloat<EXP_PUTRESULT, FRAC_PUTRESULT>((flexfloat<EXP_CND, FRAC_CND>(1.0) - CNDD1));
    *CallResult = double(ff_CallResult);
    *PutResult  = double(ff_PutResult);
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
//  print_flexfloat_stats();

  return 0;
}

