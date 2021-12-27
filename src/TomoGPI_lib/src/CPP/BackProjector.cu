/*
 * BackProjector.cu
 *
 *      Author: gac
 */

#include "BackProjector.cuh"
#include "BackProjector_CPU.cuh"
#include "BackProjector_GPU.cuh"
//#include "BackProjector_MGPU.cuh"
#include "BackProjector_kernel.cuh"

/* BackProjector definition */

template<template<typename> class V, template<typename> class S,typename T>
BackProjector<V, S, T>::BackProjector(Acquisition* acquisition, Detector* detector, V<T>* volume, char fdk) : acquisition(acquisition), detector(detector),  volume(volume), fdk(fdk)
{
	double startAngle = acquisition->getStartAngle();
	unsigned short projectionNb = acquisition->getProjectionNb();
	float focusDetectorDistance = acquisition->getFocusDetectorDistance();
	float zVolumePixelSize = volume->getZVolumePixelSize();
	float vDetectorPixelSize = detector->getVDetectorPixelSize();
	this->alphaIOcylinderC = new float[projectionNb];
	this->betaIOcylinderC = new float[projectionNb];
	this->gammaIOcylinderC = vDetectorPixelSize/(focusDetectorDistance*zVolumePixelSize);
	double* phiValueTab = acquisition->getPhiValue();

	for (int p=0;p<projectionNb;p++)
	{
		alphaIOcylinderC[p] = cos(phiValueTab[p]);
		betaIOcylinderC[p] = sin(phiValueTab[p]);
	}

	this->alphaC = (M_PI*acquisition->getFocusObjectDistance()*acquisition->getFocusDetectorDistance())/(double)(projectionNb);
	this->betaC = (acquisition->getFocusDetectorDistance()/detector->getUDetectorPixelSize())*(acquisition->getFocusDetectorDistance()/detector->getUDetectorPixelSize());
	
}

template<template<typename> class V, template<typename> class S,typename T>
BackProjector<V, S, T>::~BackProjector()
{
	delete alphaIOcylinderC;
	delete betaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
char BackProjector<V, S, T>::getFdk()
{
	return this->fdk;
}

template<template<typename> class V, template<typename> class S,typename T>
unsigned long int BackProjector<V, S, T>::getProjectionNb()
{
	return this->getAcquisition()->getProjectionNb();
}

template<template<typename> class V, template<typename> class S,typename T>
float* BackProjector<V, S, T>::getAlphaIOcylinderC()
{
	return this->alphaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
float* BackProjector<V, S, T>::getBetaIOcylinderC()
{
	return this->betaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
float BackProjector<V, S, T>::getGammaIOcylinderC()
{
	return this->gammaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
float BackProjector<V, S, T>::getAlphaC()
{
	return this->alphaC;
}

template<template<typename> class V, template<typename> class S,typename T>
float BackProjector<V, S, T>::getBetaC()
{
	return this->betaC;
}

template<template<typename> class V, template<typename> class S,typename T>
Acquisition* BackProjector<V, S, T>::getAcquisition() const
{
	return this->acquisition;
}

template<template<typename> class V, template<typename> class S,typename T>
Detector* BackProjector<V, S, T>::getDetector() const
{
	return this->detector;
}

template<template<typename> class V, template<typename> class S,typename T>
V<T>* BackProjector<V, S, T>::getVolume() const
{
	return this->volume;
}

template<template<typename> class V, template<typename> class S,typename T>
void BackProjector<V, S, T>::setAcquisition(Acquisition* acquisition)
{
	this->acquisition = acquisition;
}

template<template<typename> class V, template<typename> class S,typename T>
void BackProjector<V, S, T>::setDetector(Detector* detector)
{
	this->detector = detector;
}

template<template<typename> class V, template<typename> class S,typename T>
void BackProjector<V, S, T>::setFdk(char fdk)
{
	this->fdk = fdk;
}

template<template<typename> class V, template<typename> class S,typename T>
void BackProjector<V, S, T>::setVolume(V<T>* volume)
{
	this->volume = volume;
}

template<template<typename> class V, template<typename> class S,typename T>
S<T>* BackProjector<V, S, T>::getSinogram3D() const
{
	return this->sinogram;
}

template<template<typename> class V, template<typename> class S,typename T>
void BackProjector<V, S, T>::setSinogram3D(S<T>* sinogram)
{
	this->sinogram = sinogram;
}

#include "BackProjector_instances.cu"