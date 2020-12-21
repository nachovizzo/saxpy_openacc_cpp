// @file      saxypy.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

std::vector<float> saxpy(const std::vector<float>& x,
                         const std::vector<float>& y,
                         float a) {
  std::vector<float> z(x.size());
  std::transform(std::execution::par_unseq,
                 x.begin(),
                 x.end(),
                 y.begin(),
                 z.begin(),
                 [=](float xi, float yi) { return a * xi + yi; });
  return z;
}

int main() {
  const int N = 1 << 20;
  const float a = 1.0F;
  const std::vector<float> x(N, 1.0F);
  const std::vector<float> y(N, 1.0F);

  auto z = saxpy(x, y, a);

  float sum = std::reduce(std::execution::par, z.cbegin(), z.cend());
  std::cout << "Final result = " << sum << std::endl;
  return 0;
}
