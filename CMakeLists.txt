cmake_minimum_required(VERSION 3.1)
project(saxpy LANGUAGES C CXX)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# Build Options
option(TARGET_GPU "Enable OpenACC GPU       Support" OFF)
option(TARGET_CPU "Enable OpenACC Multicore support" OFF)

if(TARGET_CPU OR TARGET_GPU)
    find_package(OpenACC REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} -Minfo=accel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -Minfo=accel")
endif()

add_executable(saxpy_c saxpy.c)
add_executable(saxpy_cpp saxpy.cpp)
add_executable(saxpy_std_par saxpy_std_par.cpp)

if(TARGET_CPU)
    set(TARGET_ACCEL "-ta=multicore")
    set(TARGET_STDPAR "-stdpar=multicore")
elseif(TARGET_GPU)
    set(TARGET_ACCEL "-gpu=cc75")
    set(TARGET_STDPAR "-stdpar=gpu")
endif()

target_compile_options(saxpy_c PUBLIC ${TARGET_ACCEL})
target_compile_options(saxpy_cpp PUBLIC ${TARGET_ACCEL})
target_compile_options(saxpy_std_par PUBLIC ${TARGET_STDPAR})

if(TARGET_CPU)
    target_link_libraries(saxpy_std_par tbb)
elseif(TARGET_GPU)
    target_link_options(saxpy_std_par PUBLIC -stdpar=gpu)
endif()
