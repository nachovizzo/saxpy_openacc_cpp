// @file      saxypy.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <iostream>

#include "device_vector.h"

device_vector<float> saxpy(const device_vector<float>& x,
                           const device_vector<float>& y,
                           float a) {
  device_vector<float> z(x.size());
#pragma acc parallel loop
  for (int i = 0; i < y.size(); ++i) {
    z[i] = a * x[i] + y[i];
  }
  return z;
}

int main() {
  const int N = 1 << 25;
  const device_vector<float> x(N, 1.0F);
  const device_vector<float> y(N, 2.0F);
  const float a = 3.0F;

  std::cout << "Computing Saxpy on C++...\n";
  auto result = saxpy(x, y, a);
  std::cout << "Done!\n";

  // Just to check the result, make a reduction on the y array(output)
  float sum = 0;
#pragma acc parallel loop reduction(+ : sum)
  for (int i = 0; i < result.size(); ++i) {
    sum += result[i];
  }

  std::cout << "Final result = " << sum << std::endl;
  return 0;
}
