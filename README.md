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

## What is so exciting about pragma based directive parallelism?

Go and check out any modern high-performance library from the modern world,
specially those who need to heavily use the GPU's always have 2 versions of
the library functionality, one for CPU, one for GPU(Just check PyTorch,
Open3D, etc). Of course, this is time consuming to mantain and therefore costly.
What is worse is that the GPU based parallelism is usually written in CUDA
which I think is an horrible extension to the C/C++ languages, plus, it's
100% tightned to a private company, which, it is no good. On the other hand,
the CPU implementation often rely on `OpenMP` directives... but not always,
which make this CPU code highly un-efficient. Specially when comparing it
against it's CUDA counterpart.

What is so beautiful about open standards like `OpenMP` and `OpenACC` is that
allows you to express parallel code in a much nicer way, and without forcing
you to marry to any particular vendor. Nowadays, the only "good" compiler
that fully supports `OpenACC` is the one from the NVIDIA-SDK. So, we are
still on the same circle, but if we developers push for a better programming
style, the support for great compilers like `gcc` and `clang` will come in no
time. Nowadays everyone still use CUDA, so that's the problem (from my
perspective).

### Don't believe my words, look this

So, the same piece of code, only 1 module to mantain. Let's take
[saxpy.cpp](saxpy.cpp) as the victim here. You first write your code, and then
express how you would like to run this in parallel with almost no intrusion to
the original code. And then you can:

#### Run the code with a GPU

```sh
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DENABLE_OPENACC=ON ..
make all
ACC_DEVICE_TYPE=nvidia ./saxpy_cpp # Runs on your NVIDIA GPU
```

#### Run the code with a **multicore** CPU

```sh
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DENABLE_OPENACC=ON ..
make all
ACC_DEVICE_TYPE=host ./saxpy_cpp # Runs on multicore, a la OpenMP
```

#### Run the code with a **singlecore** CPU

```sh
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DENABLE_OPENACC=OFF ..
make all
./saxpy_cpp # Runs on singlecore, look at -DENABLE_OPENACC=OFF ..
```

#### Virtually with any kind of accelerator

FPGA, DSP, AMD GPU etc, Though I've never tried this myself.

## ToDo

- [ ] Add `CUDA` example, probably one from scratch and one from CUBLAS.
- [ ] Add `std::par` C++17 example
- [ ] Add pybind11 bindings
- [ ] Add Cython bindings
- [ ] Add Jupyter Notebook to benchmark all examples
