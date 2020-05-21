#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#ifndef WINDOW
#define WINDOW    1024
#endif

#ifndef CHANNELS
#define CHANNELS  4096
#endif

#ifndef FLOAT
#define FLOAT double
#endif

__device__ static const FLOAT ch_2[2] = { 0.70710678118654752440, 0.70710678118654752440 };
__device__ static const FLOAT cg_2[2] = { 0.70710678118654752440, -(0.70710678118654752440) };

#define ELEMENT(a,stride,i) ((a)[(stride)*(i)])

typedef enum
{
  gsl_wavelet_forward = 1, gsl_wavelet_backward = -1
}
gsl_wavelet_direction;

//DEF WAVELET E WORKSPACE
typedef struct
{
  const char *name;
  int (*init) (const FLOAT **h1, const FLOAT **g1,
               const FLOAT **h2, const FLOAT **g2, size_t * nc,
               size_t * offset, size_t member);
}
gsl_wavelet_type;

typedef struct
{
  //const gsl_wavelet_type *type;
  const FLOAT *h1;
  const FLOAT *g1;
  const FLOAT *h2;
  const FLOAT *g2;
  size_t nc;
  size_t offset;
}
gsl_wavelet;


typedef struct
{
  FLOAT *scratch;
  int n;
}
gsl_wavelet_workspace;

__device__
static void dwt_step (const gsl_wavelet * w, FLOAT *a, size_t stride, size_t n,
          gsl_wavelet_direction dir, gsl_wavelet_workspace * work)
{

	size_t i, ii;
	size_t jf;
	size_t k;
	size_t n1, ni, nh, nmod;

	for (i = 0; i < work->n; i++)
	{
       work->scratch[i] = 0.0f;
	}

	nmod = w->nc * n;
	nmod -= w->offset; 

	n1 = n - 1;
	nh = n >> 1;


	ii = 0;
	FLOAT h,g;

	for (i = 0; i < n; i += 2)
	{
	    h = 0;
	    g = 0;

        ni = i + nmod;

	    for (k = 0; k < w->nc; k++)
	    {
           jf = n1 & (ni + k);
           h += w->h1[k] * ((a)[(stride)*(jf)]);//ELEMENT (a, stride, jf);
           g += w->g1[k] * ((a)[(stride)*(jf)]);//ELEMENT (a, stride, jf);
           //printf("## %f %f %x\n", w->h1[k], w->g1[k],a[0]);
	    }

	    work->scratch[ii] += h;
	    work->scratch[ii + nh] += g;

           ii++;

	}

	for (i = 0; i < n; i++)
	{
       ELEMENT (a, stride, i) = work->scratch[i];
    }
    
}


__global__
void gsl_wavelet_transform (FLOAT *data, size_t stride, size_t n)
{
	int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id
	if(index >= CHANNELS) return;

	gsl_wavelet haar;
	gsl_wavelet *w;
	gsl_wavelet_workspace workspace;
	gsl_wavelet_workspace *work;
	gsl_wavelet_direction dir = gsl_wavelet_forward;
	size_t i;
	FLOAT ptr_scratch[WINDOW];

	w=&haar;

	//wavelet_alloc(w);
	w->h1=ch_2;
	w->g1=cg_2;
	w->h2=ch_2;
	w->g2=cg_2;
	w->nc=2;
	w->offset=0;

	work=&workspace;
	work->n=WINDOW;
	work->scratch=ptr_scratch;

	for (i = n; i >= 2; i >>= 1)
	{
		dwt_step (w, data+WINDOW*index, stride, i, dir, work);
	}

}


int main()
{
	int i;
	FLOAT *data;

	FLOAT *d_data;


	srand(5497);

	data = (FLOAT *)malloc(CHANNELS*WINDOW*sizeof(FLOAT));
	for(i=0; i<CHANNELS*WINDOW; i++)
		data[i] = (FLOAT)rand() / (FLOAT)RAND_MAX;
	cudaMalloc( (void **)&d_data, CHANNELS*WINDOW*sizeof(FLOAT));
	cudaMemcpy(d_data, data, CHANNELS*WINDOW*sizeof(FLOAT), cudaMemcpyHostToDevice);


    int blockSize=256;
    int numBlocks = (CHANNELS+blockSize-1)/blockSize; 

 	gsl_wavelet_transform<<<numBlocks, blockSize>>>(d_data, 1, WINDOW);

  	cudaMemcpy(data, d_data, CHANNELS*WINDOW*sizeof(FLOAT), cudaMemcpyDeviceToHost);

	for(i=0; i< CHANNELS*WINDOW; i++)
	{
    	printf("%.15f,", data[i]);
 	}

	return 0;
}
