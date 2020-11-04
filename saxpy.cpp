// @file      saxypy.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <iostream>

#include "device_vector.h"

void saxpy(const device_vector<float>& x, device_vector<float>& y, float a) {
#pragma acc parallel loop
  for (int i = 0; i < y.size(); ++i) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  const int N = 1000000;

  const device_vector<float> x(N, 1.0F);
  device_vector<float> y(N, 2.0F);
  const float a = 3.0F;

  std::cout << "Computing Saxpy on C++...\n";
  saxpy(x, y, a);
  std::cout << "Done!\n";

  // Just to check the result, make a reduction on the y array(output)
  float sum = 0;
#pragma acc parallel loop reduction(+ : sum)
  for (int i = 0; i < y.size(); ++i) {
    sum += y[i];
  }

  std::cout << "Final result = " << sum << std::endl;
  return 0;
}
