/*
 * Volume_GPU.cu
 *
 *      Author: gac
 */

#include "Volume_GPU.cuh"
#include "GPUConstant.cuh"


template <typename T>
Volume_GPU<T>::Volume_GPU() : Volume<T>(){}

template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb){}


template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture,T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb), cudaArchitecture(cudaArchitecture){}

template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_GPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture));
}

template <typename T>
Volume_GPU<T>::~Volume_GPU()
{
	delete this->getVolumeImage();
}

template <typename T>
Volume_GPU<T>::Volume_GPU(const Volume_GPU<T>& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_GPU<T>(*volumeToCopy.getVolumeImage()));
}

template <typename T>
Image3D_GPU<T>* Volume_GPU<T>::getVolumeImage() const
{
	return (Image3D_GPU<T>*)Volume<T>::getVolumeImage();
}

template <typename T>
void Volume_GPU<T>::setVolume(T value)
{
	this->getVolumeImage()->setImage(value);
}

template <typename T>
void Volume_GPU<T>::scalarVolume(T value)
{
	this->getVolumeImage()->scalarImage(value);
}

template <typename T>
void Volume_GPU<T>::addVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::addVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_GPU<T>::positiveAddVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_GPU<T>::diffVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::diffVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage(), lambda);
}

template <typename T>
void Volume_GPU<T>::diffVolume(T lambda, Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::multVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

template <typename T>
T Volume_GPU<T>::scalarProductVolume(Volume_GPU<T>* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

template <typename T>
T Volume_GPU<T>::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

template <typename T>
T Volume_GPU<T>::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

template <typename T>
T Volume_GPU<T>::getVolumeLpNorm(T p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

template <typename T>
T Volume_GPU<T>::getVolumeHuberNorm(T p)
{
	return this->getVolumeImage()->getImageHuberNorm(p);
}

template <typename T>
T Volume_GPU<T>::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

template <typename T>
T Volume_GPU<T>::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

template <typename T>
T Volume_GPU<T>::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

template <typename T>
void Volume_GPU<T>::getVolumeSign(Volume_GPU<T>* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::getVolumeAbsPow(Volume_GPU<T>* absPowVolume, T p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}





template <typename T>
void Volume_GPU<T>::grad_xplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_xmoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_yplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_ymoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_zplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_zmoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::sign_volume(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::weightVolume(T* weights){}

template <typename T>
double Volume_GPU<T>::sumWeightedVolume(T* weights){
	return 0.0;
}


Volume_GPU_half::Volume_GPU_half() : Volume_GPU<half>(){}

Volume_GPU_half::Volume_GPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume_GPU<half>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb, cudaArchitecture){}

Volume_GPU_half::~Volume_GPU_half(){}

Volume_GPU_half::Volume_GPU_half(const Volume_GPU_half& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_GPU_half(*volumeToCopy.getVolumeImage()));
}

Image3D_GPU_half* Volume_GPU_half::getVolumeImage() const
{
	return (Image3D_GPU_half*)Volume_GPU<half>::getVolumeImage();
}

void Volume_GPU_half::setVolume(float value)
{
	this->getVolumeImage()->setImage(value);
}

void Volume_GPU_half::scalarVolume(float value)
{
	this->getVolumeImage()->scalarImage(value);
}

void Volume_GPU_half::addVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

void Volume_GPU_half::addVolume(Volume_GPU_half* volume2, float lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

void Volume_GPU_half::positiveAddVolume(Volume_GPU_half* volume2, float lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

void Volume_GPU_half::diffVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

void Volume_GPU_half::diffVolume(float lambda, Volume_GPU_half* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

void Volume_GPU_half::multVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}
/*
template <>
float  Volume_GPU_half::scalarProductVolume(Volume_GPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage<float>(volume2->getVolumeImage());
}

template <>
float  Volume_GPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm<float>();
}

template <>
float  Volume_GPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm<float>();
}

template <>
float  Volume_GPU_half::getVolumeLpNorm(float p)
{
	return this->getVolumeImage()->getImageLpNorm<float>(p);
}

template <>
float  Volume_GPU_half::getVolumeHuberNorm(float threshold)
{
	return this->getVolumeImage()->getImageHuberNorm<float>(threshold);
}

template <>
float Volume_GPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean<float>();
}

template <>
float Volume_GPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare<float>();
}

template <>
float Volume_GPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd<float>();
}*/

//template <>
double Volume_GPU_half::scalarProductVolume(Volume_GPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage<double>(volume2->getVolumeImage());
}

//template <>
double Volume_GPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm<double>();
}

//template <>
double Volume_GPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm<double>();
}

//template <>
double Volume_GPU_half::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm<double>(p);
}

//template <>
double Volume_GPU_half::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm<double>(threshold);
}

//template <>
double Volume_GPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean<double>();
}

//template <>
double Volume_GPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare<double>();
}

//template <>
double Volume_GPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd<double>();
}

void Volume_GPU_half::getVolumeSign(Volume_GPU_half* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

//template <>
/*void Volume_GPU_half::getVolumeAbsPow(Volume_GPU_half* absPowVolume, float p)
{
	this->getVolumeImage()->getImageAbsPow<float>(absPowVolume->getVolumeImage(),p);
}*/

//template <>
void Volume_GPU_half::getVolumeAbsPow(Volume_GPU_half* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow<double>(absPowVolume->getVolumeImage(),p);
}

void Volume_GPU_half::saveVolume(string fileName)
{
	this->getVolumeImage()->saveImage(fileName);
}

void Volume_GPU_half::saveMiddleSliceVolume(string fileName)
{
	this->getVolumeImage()->saveMiddleSliceImage(fileName);
}

void Volume_GPU_half::loadVolume(string fileName)
{
	this->getVolumeImage()->loadImage(fileName);
}





void Volume_GPU_half::grad_xplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_xmoins(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_yplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_ymoins(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_zplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_zmoins(Volume_GPU_half* volume){}

void Volume_GPU_half::sign_volume(Volume_GPU_half* volume){}

#include "Volume_instances.cu"
#include "Volume_instances_GPU.cu"