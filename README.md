# Saxpy C vs C++ Using OpenACC

This is just a small example of how **I** think you should be using C++ over
C even when working with HPC applications, using OpenACC. This is still work
in progress and there are no guarantees.

## Why C++

Well, you better check my [C++ Course FAQ](https://www.ipb.uni-bonn.de/teaching/cpp-2020/faq/)

## What do you preffer?

You can check the 2 examples and let me know which one is more readable for you

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

```sh
benchmarking ./saxpy_c
time                 414.8 ms   (381.1 ms .. 436.2 ms)
                     0.997 R²   (0.995 R² .. 1.000 R²)
mean                 418.4 ms   (404.3 ms .. 433.7 ms)
std dev              17.27 ms   (7.404 ms .. 24.01 ms)
variance introduced by outliers: 19% (moderately inflated)
```

vs

```sh
benchmarking ./saxpy_cpp
time                 315.1 ms   (283.7 ms .. 356.1 ms)
                     0.992 R²   (0.980 R² .. 1.000 R²)
mean                 291.0 ms   (283.1 ms .. 303.9 ms)
std dev              13.81 ms   (2.627 ms .. 18.89 ms)
variance introduced by outliers: 16% (moderately inflated)
```

Anyway, I don't have anything particular against `C` per say, and if you love
it or you are still convinced that `C++` add "extra overhead" just because it
supports the word `class`, or because you don't believe in the "0 cost
abstraction" principle we all `C++` programmers believe, then, it's fine. I'm
sorry if I hurt your feelings.
