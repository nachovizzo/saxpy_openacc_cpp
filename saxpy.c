// @file      saxpy.c
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <stdio.h>
#include <stdlib.h>

void saxpy(int n, float a, const float *x, float *restrict y) {
#pragma acc kernels
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  const int N = 1 << 25;  // Huge number

  float *x = (float *)malloc(N * sizeof(float));
  float *y = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    x[i] = 2.0F;
    y[i] = 1.0F;
  }

  printf("Computing saxypy...\n");
  saxpy(N, 3.0F, x, y);
  printf("Done!\n");

  // Just to check the result, make a reduction on the y array(output)
  float sum = 0;
#pragma acc parallel loop reduction(+ : sum)
  for (int i = 0; i < N; ++i) {
    sum += y[i];
  }

  printf("Final result = %F\n", sum);
  return 0;
}
