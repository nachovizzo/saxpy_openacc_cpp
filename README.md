# Saxpy C vs C++ Using OpenACC

This is just a small example of how **I** think you should be using C++ over
C even when working with HPC applications, using OpenACC. I strongly believe
that one should do the things that would like the world to be doing, and for
me, this is one of them :).

## Why C++

Well, you better check my [C++ Course FAQ](https://www.ipb.uni-bonn.de/teaching/cpp-2020/faq/)

## What do I need to build this project?

A compiler that supports the OpenACC standard... sadly, the only one you
might be able to use is the PGI compilers(now part of the NVIDIA-HPC-SDK).

After doing that you can build this project like this(assuming you have
installed the nvidia-hpc-sdk):

```sh
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DENABLE_OPENACC=ON ..
make all
```

The output should look similar to

```sh
saxpy:
      8, Generating implicit copy(y[:n]) [if not already present]
         Generating implicit copyin(x[:n]) [if not already present]
     10, Loop is parallelizable
         Generating Tesla code
         10, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
```

## What if my compiler doesn't support OpenACC?

Well... then there is no reason for you to be here :)

## Bbbbbbbut I was told `C` is faster than `C++`

And do you still believe it? C might be faster than C++ under some particular
assumptions, one of them is that you are living in 1990 and notin 2020... if
you don't believe me you can benchmark this application by your own:

**C++ Version:**

```sh
time                 121.0 ms   (98.66 ms .. 149.2 ms)
                     0.953 R²   (0.855 R² .. 0.999 R²)
mean                 138.5 ms   (130.5 ms .. 152.5 ms)
std dev              16.58 ms   (8.188 ms .. 24.11 ms)
variance introduced by outliers: 36% (moderately inflated)
```

**C Version:**

```sh
benchmarking ./saxpy_c
time                 134.0 ms   (113.3 ms .. 157.7 ms)
                     0.962 R²   (0.891 R² .. 0.997 R²)
mean                 130.3 ms   (122.9 ms .. 139.2 ms)
std dev              12.42 ms   (8.789 ms .. 17.49 ms)
variance introduced by outliers: 24% (moderately inflated)
```

Anyway, I don't have anything particular against `C` per say, and if you love
it or you are still convinced that `C++` add "extra overhead" just because it
supports the word `class`, or because you don't believe in the "0 cost
abstraction" principle we all `C++` programmers believe, then, it's fine. I'm
sorry if I hurt your feelings.

## ToDO

 - [ ] Add `CUDA` example, probably one from scratch and one from CUBLAS.
 - [ ] Add `std::par` C++17 example
 - [ ] Add pybind11 bindings
 - [ ] Add Cython bindings
 - [ ] Add Jupyter Notebook to benchmark all examples