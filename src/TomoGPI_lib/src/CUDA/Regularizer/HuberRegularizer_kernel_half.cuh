/*
 * HuberRegularizer_kernel.cu
 *
 *      Author: gac
 */

#ifndef _HUBERREGULARIZER_KERNEL_HALF_H_
#define _HUBERREGULARIZER_KERNEL_HALF_H_



__global__ void derivativeHuberFunction_k_half(unsigned short* volume, unsigned short* derivativeHuberVolume, float huberThres, unsigned long int n)
{

	unsigned int xn,yn,zn,id;
	float threshold = huberThres;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	float tmp = 0;
	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(volume[id]);
		if(tmp <= -1.0*threshold)
			derivativeHuberVolume[id] = __float2half_rn(-2.0*threshold);
		else if (fabs(tmp) < threshold)
			derivativeHuberVolume[id] = __float2half_rn(2.0*tmp);
		else if (tmp >= threshold)
			derivativeHuberVolume[id] = __float2half_rn(2.0*threshold);
		else
			derivativeHuberVolume[id] = __float2half_rn(tmp);
	}
}

__global__ void secondDerivativeHuberFunction_k_half(unsigned short* volume, unsigned short* secondDerivativeHuberVolume, float huberThres, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	float threshold = huberThres;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	float tmp = 0;
	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	if (id < n)
	{
		tmp = __half2float(volume[id]);
		if(fabs(tmp) > threshold)
			secondDerivativeHuberVolume[id] = __float2half_rn(0.0);
		else if (fabs(tmp) <= threshold)
			secondDerivativeHuberVolume[id] = __float2half_rn(1.0);
		else
			secondDerivativeHuberVolume[id] = __float2half_rn(tmp);
	}
}

#endif
