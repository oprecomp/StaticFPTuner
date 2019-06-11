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

#include <stdio.h>

#include "data.h"

#include "flexfloat.hpp"

void saxpy(int n, double a, double * __restrict x, double * __restrict y)
{
  for (int i = 0; i < n; ++i)
      y[i] = (double) (flexfloat<EXP_Y, FRAC_Y>(flexfloat<EXP_A, FRAC_A>(a))*flexfloat<EXP_Y, FRAC_Y>(flexfloat<EXP_X, FRAC_X>(x[i])) + flexfloat<EXP_Y, FRAC_Y>(y[i]));
}

int main()
{
  saxpy(SIZE, 2.0, input, output);  
  int i;
  for(i=0; i<SIZE; ++i)
    printf("%.15f,", output[i]);
  //print_flexfloat_stats();
  return 0;
}
