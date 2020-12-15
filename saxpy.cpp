// @file      saxypy.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <cassert>
#include <iostream>

#include "device_vector.h"

device_vector<float> saxpy(const device_vector<float>& x,
                           const device_vector<float>& y,
                           float a) {
  assert(x.size() == y.size());
  device_vector<float> z(x.size());
  #pragma acc parallel loop
  for (int i = 0; i < y.size(); ++i) {
    z[i] = a * x[i] + y[i];
  }
  return z;
}

int main() {
  const int N = 1 << 20;
  const float a = 1.0F;
  const device_vector<float> x(N, 1.0F);
  const device_vector<float> y(N, 1.0F);

  auto z = saxpy(x, y, a);
  float sum = 0;
  #pragma acc parallel loop reduction(+ : sum)
  for (int i = 0; i < z.size(); ++i) {
    sum += z[i];
  }

  std::cout << "Final result = " << sum << std::endl;
  return 0;
}
