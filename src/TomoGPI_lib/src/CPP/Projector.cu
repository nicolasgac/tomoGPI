/*
 * Projector.cu
 *
 * Author: gac
 */

#include "Projector.cuh"
#include "Projector_CPU.cuh"
#include "Projector_GPU.cuh"
//#include "Projector_MGPU.cuh"
#include "Projector_kernel.cuh"

#include "projection_sftr_mergedDir_kernel.cuh"
#include "weightedCoeffDiagHVHT_sftr_kernel.cuh"
#include "projection_sftr_opti.cuh"
#include "projection_sf_rec_rec_kernel.cuh"


template<template<typename> class V, template<typename> class S,typename T>
Projector<V,S,T>::Projector(Acquisition* acquisition, Detector* detector, V<T>* volume) : acquisition(acquisition), detector(detector),  volume(volume){

	float startAngle = acquisition->getStartAngle();
	float focusDetectorDistance = acquisition->getFocusDetectorDistance();
	float zVolumePixelSize = volume->getZVolumePixelSize();
	float vDetectorPixelSize = detector->getVDetectorPixelSize();


	unsigned short projectionNb = acquisition->getProjectionNb();
	this->alphaIOcylinderC = new float[projectionNb];
	this->betaIOcylinderC = new float[projectionNb];
	this->gammaIOcylinderC = vDetectorPixelSize/(focusDetectorDistance*zVolumePixelSize);

	double* phiValueTab = acquisition->getPhiValue();

	for (int p=0;p<projectionNb;p++)
	{
		this->alphaIOcylinderC[p] = cos(phiValueTab[p]);
		this->betaIOcylinderC[p] = sin(phiValueTab[p]);
	}


}


/*
template<template<typename> class V, template<typename> class S,typename T>
Projector<V,S,T>::Projector() : acquisition(0), detector(0),  cudaprojectionArchitecture(0),volume(0){}
 */

template<template<typename> class V, template<typename> class S,typename T>
Projector<V,S,T>::~Projector(){
	delete alphaIOcylinderC;
	delete betaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
Acquisition* Projector<V,S,T>::getAcquisition() const
{
	return this->acquisition;
}

template<template<typename> class V, template<typename> class S,typename T>
Detector* Projector<V,S,T>::getDetector() const
{
	return this->detector;
}

template<template<typename> class V, template<typename> class S,typename T>
float* Projector<V,S,T>::getAlphaIOcylinderC()
{
	return this->alphaIOcylinderC;	
}

template<template<typename> class V, template<typename> class S,typename T>
float* Projector<V,S,T>::getBetaIOcylinderC()
{
	return this->betaIOcylinderC;
}

template<template<typename> class V, template<typename> class S,typename T>
float Projector<V,S,T>::getGammaIOcylinderC()
{
	return this->gammaIOcylinderC;
}



template<template<typename> class V, template<typename> class S,typename T>
V<T>* Projector<V,S,T>::getVolume() const
{
	return this->volume;
}

template<template<typename> class V, template<typename> class S,typename T>
void Projector<V,S,T>::setAcquisition(Acquisition* acquisition)
{
	this->acquisition = acquisition;
}

template<template<typename> class V, template<typename> class S,typename T>
void Projector<V,S,T>::setDetector(Detector* detector)
{
	this->detector = detector;
}


template<template<typename> class V, template<typename> class S,typename T>
void Projector<V,S,T>::setVolume(V<T>* volume)
{
	this->volume = volume;
}

template<template<typename> class V, template<typename> class S,typename T>
void Projector<V,S,T>::updateVolume(V<T>* volume_out,V<T>* volume_in, double lambda, int positive)
{

	//volume_out->saveVolume("test_out.v");
	//volume_in->saveVolume("test_in.v");

	if(positive)
	{
		volume_out->positiveAddVolume(volume_in, lambda);
	}
	else
	{
		volume_out->addVolume(volume_in, lambda);
	}
}

#include "Projector_instances_CPU.cu"
#include "Projector_instances_GPU.cu"
//#include "Projector_instances_MGPU.cu"