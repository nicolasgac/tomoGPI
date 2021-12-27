/*
 * Convolution3D.cu
 *
 *      Author: gac
 */

#include "Convolution3D.cuh"
#include "GPUConstant.cuh"


template <typename T>
Convolution3D<T>::Convolution3D(Image3D<T>* kernel): kernel(kernel) , horizontalKernel(0), verticalKernel(0), depthKernel(0)
{
	this->xKernelRadius = (kernel->getXImagePixelNb()-1)/2;
	this->yKernelRadius = (kernel->getYImagePixelNb()-1)/2;
	this->zKernelRadius = (kernel->getZImagePixelNb()-1)/2;
	this->xKernelSize = (this->kernel)->getXImagePixelNb();
	this->yKernelSize = (this->kernel)->getYImagePixelNb();
	this->zKernelSize = (this->kernel)->getZImagePixelNb();
}

template <typename T>
Convolution3D<T>::Convolution3D(T* horizontalKernel, T* verticalKernel, T* depthKernel): horizontalKernel(horizontalKernel), verticalKernel(verticalKernel), depthKernel(depthKernel), kernel(0)
{
	this->xKernelRadius = 1;
	this->yKernelRadius = 1;
	this->zKernelRadius = 1;
	this->xKernelSize = 2*this->xKernelRadius+1;
	this->yKernelSize = 2*this->yKernelRadius+1;
	this->zKernelSize = 2*this->zKernelRadius+1;
}

template <typename T>
Convolution3D<T>::~Convolution3D(){}

template <typename T>
Image3D<T>* Convolution3D<T>::getKernel() const
{
	return this->kernel;
}

template <typename T>
unsigned long int Convolution3D<T>::getXKernelRadius() const
{
	return this->xKernelRadius;
}

template <typename T>
unsigned long int Convolution3D<T>::getYKernelRadius() const
{
	return this->yKernelRadius;
}

template <typename T>
unsigned long int Convolution3D<T>::getZKernelRadius() const
{
	return this->zKernelRadius;
}

template <typename T>
unsigned long int Convolution3D<T>::getXKernelSize() const
{
	return this->xKernelSize;
}

template <typename T>
unsigned long int Convolution3D<T>::getYKernelSize() const
{
	return this->yKernelSize;
}

template <typename T>
unsigned long int Convolution3D<T>::getZKernelSize() const
{
	return this->zKernelSize;
}

template <typename T>
unsigned long int Convolution3D<T>::getXFrameSize(Image3D<T>* sourceImage) const
{
	return 	sourceImage->getXImagePixelNb() + 2*this->xKernelRadius;
}

template <typename T>
unsigned long int Convolution3D<T>::getYFrameSize(Image3D<T>* sourceImage) const
{
	return 	sourceImage->getYImagePixelNb() + 2*this->yKernelRadius;
}

template <typename T>
unsigned long int Convolution3D<T>::getZFrameSize(Image3D<T>* sourceImage) const
{
	return 	sourceImage->getZImagePixelNb() + 2*this->zKernelRadius;
}

/* Copy convolution constant */
template <typename T>
__host__ void Convolution3D<T>::copyConstantGPU(Volume<T>* sourceImage)
{
	const unsigned long int xVolumeSize	= sourceImage->getXVolumePixelNb();
	const unsigned long int yVolumeSize = sourceImage->getYVolumePixelNb();
	const unsigned long int zVolumeSize = sourceImage->getZVolumePixelNb();

	cudaMemcpyToSymbol(c_Kernel_h, this->horizontalKernel, this->xKernelSize*sizeof(T));
	cudaMemcpyToSymbol(c_Kernel_v, this->verticalKernel, this->yKernelSize*sizeof(T));
	cudaMemcpyToSymbol(c_Kernel_p, this->depthKernel, this->zKernelSize*sizeof(T));

	cudaMemcpyToSymbol(c_volume_x, &xVolumeSize, sizeof(unsigned long int));
	cudaMemcpyToSymbol(c_volume_y, &yVolumeSize, sizeof(unsigned long int));
	cudaMemcpyToSymbol(c_volume_z, &zVolumeSize, sizeof(unsigned long int));

	cudaMemcpyToSymbol(c_kernel_radius_x, &(this->xKernelRadius), sizeof(unsigned long int));
	cudaMemcpyToSymbol(c_kernel_radius_y, &(this->yKernelRadius), sizeof(unsigned long int));
	cudaMemcpyToSymbol(c_kernel_radius_z, &(this->zKernelRadius), sizeof(unsigned long int));
}

#include "Convolution3D_instances.cu"

