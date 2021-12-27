/*
 * Convolution3D_GPU.cuh
 *
 *      Author: gac
 */

#ifndef CONVOLUTION3D_GPU_HPP_
#define CONVOLUTION3D_GPU_HPP_

#include "Convolution3D.cuh"
#include "Volume_GPU.cuh"


template<typename T> class Convolution3D_GPU : public Convolution3D<T>{
public:
	Convolution3D_GPU(Image3D_GPU<T>* kernel); // Constructor for simple 3D convolution
	Convolution3D_GPU(T* horizontalKernel, T* verticalKernel, T* depthKernel); // Constructor for separable 3D convolution
	~Convolution3D_GPU();
	/* Separable 3D Convolution on GPU */
	void doSeparableConvolution3D(Volume_GPU<T>* sourceImage, Volume_GPU<T>* convolutedImage);
};

class Convolution3D_GPU_half : public Convolution3D_GPU<float>{
public:
	Convolution3D_GPU_half(Image3D_GPU<float>* kernel); // Constructor for simple 3D convolution
	Convolution3D_GPU_half(float* horizontalKernel, float* verticalKernel, float* depthKernel); // Constructor for separable 3D convolution
	~Convolution3D_GPU_half();
	/* Separable 3D Convolution on GPU */
	void doSeparableConvolution3D(Volume_GPU_half* sourceImage, Volume_GPU_half* convolutedImage);
};

#endif /* CONVOLUTION3D_HPP_ */
