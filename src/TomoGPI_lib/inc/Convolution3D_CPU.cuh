/*
 * Convolution3D_CPU.cuh
 *
 *      Author: gac
 */

#ifndef CONVOLUTION3D_CPU_HPP_
#define CONVOLUTION3D_CPU_HPP_

#include "Convolution3D.cuh"
#include "Volume_CPU.cuh"

template<typename T> class Convolution3D_CPU : public Convolution3D<T>{
public:
	Convolution3D_CPU(Image3D_CPU<T>* kernel); // Constructor for simple 3D convolution
	Convolution3D_CPU(T* horizontalKernel, T* verticalKernel, T* depthKernel); // Constructor for separable 3D convolution
	~Convolution3D_CPU();
	static CUT_THREADPROC solverThread(TGPUplan_conv3D<Volume_CPU, T > *plan);
	/* Separable 3D Convolution on GPU */
	void doSeparableConvolution3D(Volume_CPU<T>* sourceImage, Volume_CPU<T>* convolutedImage);
};

class Convolution3D_CPU_half : public Convolution3D_CPU<float>{
public:
	Convolution3D_CPU_half(Image3D_CPU<float>* kernel); // Constructor for simple 3D convolution
	Convolution3D_CPU_half(float* horizontalKernel, float* verticalKernel, float* depthKernel); // Constructor for separable 3D convolution
	~Convolution3D_CPU_half();
	static CUT_THREADPROC solverThread(TGPUplan_conv3D_half<Volume_CPU_half> *plan);
	/* Separable 3D Convolution on GPU */
	void doSeparableConvolution3D(Volume_CPU_half* sourceImage, Volume_CPU_half* convolutedImage);
};

#endif /* CONVOLUTION3D_HPP_ */
