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
  const int N = 1 << 25;
  const device_vector<float> x(N, 2.0F);
  device_vector<float> y(N, 1.0F);

  std::cout << "Computing Saxpy on C++...\n";
  saxpy(x, y, 3.0F);
  std::cout << "Done!\n";
  return 0;
}
