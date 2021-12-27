/*
 * half_float_conversion_kernel.cu
 *
 *  Created on: 2 aout 2018
 *      Author: gac
 */
#ifndef _HALF_FLOAT_CONVERSION_KERNEL_CU_
#define _HALF_FLOAT_CONVERSION_KERNEL_CU_

#include <helper_functions.h>
#include <helper_cuda.h>
#include <multithreading.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>

__global__ void half_to_float_kernel(half *data_16f,float *data_32f);
#endif
