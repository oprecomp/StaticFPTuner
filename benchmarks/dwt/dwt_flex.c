#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flexfloat.hpp"

#define WINDOW    256
#define CHANNELS  13

#ifndef FLOAT
#define FLOAT double
#endif

#include "datasets.h"

static const FLOAT ch_2[2] = { 0.70710678118654752440, 0.70710678118654752440 };
static const FLOAT cg_2[2] = { 0.70710678118654752440, -(0.70710678118654752440) };

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


//#define ERROR_TRACKING

#ifdef ERROR_TRACKING
void callback(flexfloat_t *v, void *arg) {
    long index = (long)arg;
    printf("[%d] exact value = %f, absolute error = %f \n ", index, ff_track_get_exact(v), ff_track_get_error(v));
    if (ff_track_get_exact(v) > 0.000001 && fabs(ff_track_get_error(v))/ff_track_get_exact(v) > 0.60) abort();
}
#endif


static void
dwt_step (const gsl_wavelet * w, FLOAT *a, size_t stride, size_t n,
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
	nmod -= w->offset;            /* center support */

	n1 = n - 1;
	nh = n >> 1;


	ii = 0;

	flexfloat<EXP_WAVELET_DS_H, FRAC_WAVELET_DS_H> h;
	flexfloat<EXP_WAVELET_DS_G, FRAC_WAVELET_DS_G> g;
  flexfloat<EXP_WAVELET_DS_A, FRAC_WAVELET_DS_A> ff_a;
  flexfloat<EXP_WAVELET_DS_H1, FRAC_WAVELET_DS_H1> ff_h1;
  flexfloat<EXP_WAVELET_DS_G1, FRAC_WAVELET_DS_G1> ff_g1;
  flexfloat<EXP_WAVELET_DS_TEMP1, FRAC_WAVELET_DS_TEMP1> temp1;
  flexfloat<EXP_WAVELET_DS_TEMP2, FRAC_WAVELET_DS_TEMP2> temp2;

#ifdef ERROR_TRACKING
  h.setCallback(callback, (void*)0);
  g.setCallback(callback, (void*)1);
#endif

	//flexfloat_vectorization = true;
	for (i = 0; i < n; i += 2)
	{
	    h = 0;
	    g = 0;

        ni = i + nmod;


	    for (k = 0; k < w->nc; k++)
	    {
           jf = n1 & (ni + k);

           ff_a = a[(stride)*(jf)];  // [/SKIP]
           ff_h1 = w->h1[k];         // [/SKIP]
           ff_g1 = w->g1[k];         // [/SKIP]



#ifdef ERROR_TRACKING
           temp1.setCallback(callback, (void*)5);
           temp2.setCallback(callback, (void*)6);
#endif

           temp1 = flexfloat<EXP_WAVELET_DS_TEMP1, FRAC_WAVELET_DS_TEMP1>(ff_h1) * flexfloat<EXP_WAVELET_DS_TEMP1, FRAC_WAVELET_DS_TEMP1>(ff_a);
           h = h + flexfloat<EXP_WAVELET_DS_H, FRAC_WAVELET_DS_H>(temp1);
           temp2 = flexfloat<EXP_WAVELET_DS_TEMP2, FRAC_WAVELET_DS_TEMP2>(ff_g1) * flexfloat<EXP_WAVELET_DS_TEMP2, FRAC_WAVELET_DS_TEMP2>(ff_a);
           g = g + flexfloat<EXP_WAVELET_DS_G, FRAC_WAVELET_DS_G>(temp2);
	    }

	    work->scratch[ii] = FLOAT(flexfloat<EXP_WAVELET_DS_H, FRAC_WAVELET_DS_H>(work->scratch[ii]) + h);
	    work->scratch[ii + nh] = FLOAT(flexfloat<EXP_WAVELET_DS_G, FRAC_WAVELET_DS_G>(work->scratch[ii + nh]) + g);

      ii++;

	}
	//flexfloat_vectorization = false;

	for (i = 0; i < n; i++)
	{
		ELEMENT (a, stride, i) = work->scratch[i];
	}

}


int
gsl_wavelet_transform (FLOAT *data, size_t stride, size_t n)
{
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
		dwt_step (w, data, stride, i, dir, work);
	}

}

int main()
{
	int i, j;

	for(i=0; i< CHANNELS; i++)
	{
 		gsl_wavelet_transform (data[i], 1, WINDOW);
		for(j=0; j<WINDOW; j++)
	    	printf("%.15f,", data[i][j]);
 	}

//print_flexfloat_stats();

	return 0;
}
