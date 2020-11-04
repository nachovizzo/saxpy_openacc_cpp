// @file      saxpy.c
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <stdio.h>
#include <stdlib.h>

void saxpy(int n, float a, const float *x, const float *y, float *restrict z) {
#pragma acc parallel loop
  for (int i = 0; i < n; ++i) {
    z[i] = a * x[i] + y[i];
  }
}

int main() {
  const int N = 1000000;

  float *x = (float *)malloc(N * sizeof(float));
  float *y = (float *)malloc(N * sizeof(float));
  float *z = (float *)malloc(N * sizeof(float));
  const float a = 3.0F;

  for (int i = 0; i < N; ++i) {
    x[i] = 1.0F;
    y[i] = 2.0F;
    z[i] = 0.0F;
  }

  printf("Computing saxypy...\n");
  saxpy(N, a, x, y, z);
  printf("Done!\n");

  // Just to check the result, make a reduction on the y array(output)
  float sum = 0;
#pragma acc parallel loop reduction(+ : sum)
  for (int i = 0; i < N; ++i) {
    sum += z[i];
  }

  printf("Final result = %d\n", (int)sum);
  return 0;
}
