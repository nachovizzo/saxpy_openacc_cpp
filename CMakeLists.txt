cmake_minimum_required(VERSION 3.1)
project(saxpy LANGUAGES C CXX)

option(ENABLE_OPENACC "Enable OpenACC" ON)
if(ENABLE_OPENACC)
  find_package(OpenACC REQUIRED)
  set(OpenACC_ACCEL_TARGET "tesla,managed,multicore")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} -Minfo=accel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -Minfo=accel")
endif()

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(saxpy_c saxpy.c)
add_executable(saxpy_cpp saxpy.cpp)
