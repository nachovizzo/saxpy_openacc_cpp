// @file      device_vector.h
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <cstddef>

// clang-format off
template <typename T> class device_vector {

public:

  device_vector() = default;

  explicit device_vector(size_t size) {
    _size = size;
    _A = new T[_size];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(_A [0:_size])
  }

  explicit device_vector(size_t size, const T &value) {
    _size = size;
    _A = new T[_size];
    for (int i = 0; i < _size; ++i) {
      _A[i] = value;
    }
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(_A [0:_size])
  }

  ~device_vector() {
    #pragma acc exit data delete (_A [0:_size])
    #pragma acc exit data delete (this)
    delete[] _A;
    _A = nullptr;
    _size = 0;
  }

  #pragma acc routine seq
  T &operator[](size_t idx) { return _A[idx]; }

  #pragma acc routine seq
  const T &operator[](size_t idx) const { return _A[idx]; }

  size_t size() const { return _size; }

private:
  T *_A{nullptr};
  size_t _size{0};
};

// clang-format on
