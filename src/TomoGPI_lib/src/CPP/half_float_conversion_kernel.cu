/*
 * half_float_conversion_kernel.cu
 *
 *      Author: gac
 */


#include "half_float_conversion_kernel.cuh"


__global__ void half_to_float_kernel(half *data_16f,float *data_32f)
{
	unsigned int xn,yn,zn;
	unsigned long long int id;

	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z + blockIdx.z*blockDim.z;


	id=xn+yn*blockDim.x * gridDim.x+zn*blockDim.x * gridDim.x*blockDim.y * gridDim.y;

	data_32f[id]=__half2float(data_16f[id]);



}

