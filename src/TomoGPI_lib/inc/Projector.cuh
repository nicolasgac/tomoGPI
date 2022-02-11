/*
 * Projector.cuh
 *
 *      Author: gac
 */

#ifndef PROJECTOR_HPP_
#define PROJECTOR_HPP_

#ifdef __linux__
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#endif
#include <omp.h>

#include "Acquisition.hpp"
#include "Detector.hpp"
#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"
#include "ComputingArchitecture.cuh"

template<template<typename> class V, template<typename> class S, typename T> struct  TGPUplan_proj{
	//! Carte GPU actuelle
	int device;
	//! Le volume
	V<T>* volume_h;
	//! Le sinogramme
	S<T>* sinogram_h;
	// Computing Architecture
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
	//! Angle de initialisation (non utilise)
	int phi_start;
	int vn_start;
	int zn_start;
	int N_zn_par_solverthread;
	int nstreams;
	unsigned long long int size_sinogram;
	dim3 dimBlock;
	dim3 dimGrid;
	unsigned int SM;
	unsigned int numberCells;
	unsigned int N_vn_par_carte;
	unsigned int N_vn_par_solverthread;
	unsigned int N_vn_par_kernel;
	unsigned int N_ligne_par_carte;
	unsigned int N_vn_restant;
	Acquisition* acquisition;
	Detector* detector;

	// MGPU
    struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	size_t SinoSize_d;
	T** Sinogram_d;

	//added
	int z0_min;
	int z0_max;
	int gpuNb;
};

template<typename V, typename S> struct  TGPUplan_proj_half{
	//! Carte GPU actuelle
	int device;
	//! Le volume
	V* volume_h;
	//! Le sinogramme
	S* sinogram_h;
	// Computing Architecture
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
	//! Angle de initialisation (non utilise)
	int phi_start;
	int vn_start;
	int zn_start;
	int N_zn_par_solverthread;
	int nstreams;
	dim3 dimBlock;
	dim3 dimGrid;
	unsigned long long int size_sinogram;
	unsigned int N_vn_par_carte;
	unsigned int N_vn_par_solverthread;
	unsigned int N_vn_par_kernel;
	unsigned int N_ligne_par_carte;
	unsigned int N_vn_restant;
	Acquisition* acquisition;
	Detector* detector;
};

#ifdef __CUDACC__
// declare texture reference for 3D float texture POUR ER
extern texture<float, cudaTextureType3D, cudaReadModeElementType> volume_tex;

//// declare texture reference for 3D float texture POUR JOSEPH
//texture<float, 2, cudaReadModeElementType> volume_tex0;
//texture<float, 2, cudaReadModeElementType> volume_tex1;
//texture<float, 2, cudaReadModeElementType> volume_tex2;
//texture<float, 2, cudaReadModeElementType> volume_tex3;
//texture<float, 2, cudaReadModeElementType> volume_tex4;
//texture<float, 2, cudaReadModeElementType> volume_tex5;
//texture<float, 2, cudaReadModeElementType> volume_tex6;
//texture<float, 2, cudaReadModeElementType> volume_tex7;
#endif


template<template<typename> class V, template<typename> class S, typename T> class Projector{
public:
	Projector(Acquisition* acquisition, Detector* detector, V<T>* volume);
	//Projector();
		virtual ~Projector();
	virtual void doProjection(S<T>* estimatedSinogram,V<T> *volume) = 0;

	Acquisition* getAcquisition() const; // Get detector
	Detector* getDetector() const; // Get detector
	V<T>* getVolume() const; // Get volume
	float * getAlphaIOcylinderC();
	float *getBetaIOcylinderC();
	float getGammaIOcylinderC();
	void setAcquisition(Acquisition* acquisition); // Set detector
	void setDetector(Detector* detector); // Set detector
	void setVolume(V<T>* volume); // Set volume
	void updateVolume(V<T>* volume_out,V<T>* volume_in, double lambda, int positive = 0); // Update volume

private:
	Acquisition* acquisition;
	Detector* detector;
	V<T>* volume;
	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	float gammaIOcylinderC; // Lambda input-output cylinder point constant
};


template<typename V, typename S> class Projector_half{
public:

	Projector_half(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,V* volume);
	Projector_half();
	virtual ~Projector_half();
	virtual void doProjection(S* estimatedSinogram,V* volume) = 0;

	Acquisition* getAcquisition() const; // Get detector
	Detector* getDetector() const; // Get detector
	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const; // Get detector
	V* getVolume() const; // Get volume
	void setAcquisition(Acquisition* acquisition); // Set detector
	void setDetector(Detector* detector); // Set detector
	void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set detector
	void setVolume(V* volume); // Set volume
	void updateVolume(V* volume_out,V* volume_in, double lambda, int positive = 0); // Update volume

#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif

private:
	Acquisition* acquisition;
	Detector* detector;
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
	V* volume;
	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	float gammaIOcylinderC; // Lambda input-output cylinder point constant
};

//template<template<typename> class V, template<typename> class S, typename T> class JosephProjector : public Projector<V,S,T>{
//
//public:
//
//	JosephProjector(Acquisition* acquisition, Detector* detector, V<T>* volume);
//	~JosephProjector();
//	void doProjection_CPU_GPU(S<T>* estimatedSinogram, CUDAProjectionArchitecture* cudaprojectionArchitecture);
//	void doProjection_GPU(S<T>* estimatedSinogram, CUDAProjectionArchitecture* cudaprojectionArchitecture);
//
//
//#ifdef __CUDACC__
//	__host__ void  copyConstantGPU();
//#endif
//
//
//private:
//	float *alphaPreComputingC; // Alpha pre-computing constant tab
//	float *betaPreComputingC; // Beta pre-computing constant tab
//	float *deltaPreComputingC; //! Delta pre-computing constant tab
//	float *sigmaPreComputingC; //! Sigma pre-computing constant tab
//	float *kappaPreComputingC; //! Kappa pre-computing constant tab
//	float *iotaPreComputingC; //! Iota pre-computing constant tab
//	float gammaPrecomputingC; //! Gamma pre-computing constant
//	float omegaPrecomputingC; //! Omega pre-computing constant
//};

#endif /* PROJECTOR_HPP_ */
