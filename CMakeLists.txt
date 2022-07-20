#cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_30” ..
cmake_minimum_required(VERSION 3.8)

project(TomoGPI_lib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/TomoGPI_lib")
set(EXE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/TomoGPI_exe")
set(MEX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/TomoGPI_mex/")

#find_package(OpenCL REQUIRED)

find_path(OpenCL_INCLUDE_DIR
  NAMES
    CL/cl.h OpenCL/cl.h
    NO_DEFAULT_PATH
  PATHS
    /usr/local/cuda/
  PATH_SUFFIXES
    include
    ${ROOT_DIR}/common/inc)

find_package(Matlab)

list(APPEND HEADER_REP "common/inc" ${ROOT_DIR}/inc ${ROOT_DIR}/inc/Config "${ROOT_DIR}/inc/GPUConstant" "${ROOT_DIR}inc/Regularizer" "${ROOT_DIR}/inc/CUDABackProjector" "${ROOT_DIR}/inc/Convolution" "${ROOT_DIR}/inc/Image3D_GPU" "${ROOT_DIR}/inc/Projector" "${ROOT_DIR}/inc/Regularizer" "${ROOT_DIR}/inc/Sinogram3D_GPU" "${ROOT_DIR}/inc/Segmentation" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc"  ${ROOT_DIR}/src/CUDA/BackProjector ${ROOT_DIR}/src/CUDA/Convolution ${ROOT_DIR}/src/CUDA/Image3D_GPU ${ROOT_DIR}/src/CUDA/Projector ${ROOT_DIR}/src/CUDA/Regularizer ${ROOT_DIR}/src/CUDA/Segmentation ${ROOT_DIR}/src/CUDA/Sinogram3D_GPU ${ROOT_DIR}/src/OCL/) #${OpenCL_INCLUDE_DIR} 

list(APPEND HEADER_REP "$ENV{CUDA_SAMPLES}/common/inc" "$ENV{INTELFPGAOCLSDKROOT}/host/include" "$ENV{INTELFPGAOCLSDKROOT}/examples_aoc/common/inc")

#list(APPEND HEADER_FILES ${ROOT_DIR}/inc/Acquisition.hpp)
#list(APPEND CPP_FILES ${ROOT_DIR}/src/CPP/Acquisition.cpp)

file(GLOB CPP_FILES ${ROOT_DIR}/src/CPP/*.cpp ${ROOT_DIR}/src/CPP/Config/*.cpp)# ${ROOT_DIR}/src/CPP/src/Config/*.cpp)
file(GLOB CU_FILES ${ROOT_DIR}/src/CPP/*.cu)
file(GLOB_RECURSE HEADER_FILES ${ROOT_DIR}/src/CPP/*.h)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math ")

add_library(TomoGPI ${CPP_FILES} ${CU_FILES} ${HEADER_FILES})

set_target_properties(TomoGPI PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(TomoGPI PUBLIC cxx_std_11)
target_include_directories(TomoGPI PUBLIC "${HEADER_REP}")

add_executable(TomoGPI_exe "${EXE_DIR}/TomoGPI_exe.cu")
target_link_libraries(TomoGPI_exe PUBLIC TomoGPI OpenCL)

#matlab_add_mex(NAME TomoGPI_iter_mex SRC ${MEX_DIR}/Mex.cu ${MEX_DIR}/TomoGPI_iter_mex.cu  "${MEX_DIR}/Mex.cuh" LINK_TO TomoGPI OpenCL)
#set_target_properties(TomoGPI_iter_mex PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_compile_options(TomoGPI_iter_mex PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler)
#target_compile_features(TomoGPI_iter_mex PUBLIC cxx_std_11)
#target_include_directories(TomoGPI_iter_mex PUBLIC "${HEADER_REP}" "${MEX_DIR}")
