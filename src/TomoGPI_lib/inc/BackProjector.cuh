/*
 * BackProjector.cuh
 *
 *      Author: gac
 */

#ifndef BACKPROJECTOR_HPP_
#define BACKPROJECTOR_HPP_

#include <cstdlib>
#ifdef __linux__
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#endif
#include <iostream>



#include "Detector.hpp"
#include "Acquisition.hpp"
#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"
#include "ComputingArchitecture.cuh"


typedef struct struct_sampling_opencl {
	//! Taille du voxel du volume en x
	float delta_xn; 
  
	float xn_0;
	//! (N_yn_FOV / 2) - 0.5
	float yn_0;
	//! (N_zn_FOV / 2) - 0.5
	float zn_0;   ///15
  
	int N_un;
	//!  Nombre de pixels du plan detecteur en v
	int N_vn;
  
	int N_xn_FOV;
  } type_struct_sampling_opencl;
  
  #define UN_MAX 1024
  #define VN_MAX 1024
  #define PHI_MAX 1024
  #define XN_MAX 1024
  #define YN_MAX 1024
  #define ZN_MAX 1024

  typedef struct struct_constante_opencl {
	float alpha_wn[PHI_MAX];
	float beta_wn[PHI_MAX];
	float gamma_wn;
	float gamma_vn;
	float D;
	float delta_un;
	float gamma_D;//20
	float un_0;
	float vn_0;
  }type_struct_constante_opencl;
  

template<template<typename> class V, template<typename> class S,typename T> struct  TGPUplan_retro{
	//! Carte GPU actuelle
	char fdk;
	int device;
	//! Le volume
	V<T>* volume_h;
	//! Le sinogramme
	S<T>* sinogram_h;
	// Computing Architecture
	CUDABProjectionArchitecture* cudabackprojectionArchitecture;
	//! nombre d'angle calculé pour de gros sinogrammes
	int N_phi_reduit;
	//! Angle de initialisation (non utilise)
	int phi_start;
	int vn_start;
	int zn_start;
	int N_zn_par_carte;
	int N_zn_par_solverthread;
	int numberVoxels;
	int N_zn_par_kernel;
	unsigned int N_vn_par_solverthread;
	Acquisition* acquisition;
	Detector* detector;

    struct cudaExtent sino_cu_3darray_size;
	cudaArray* sino_cu_3darray;

	size_t VolSize_d;
	T* Volume_d;


	//added
	int v0_min;
	int v0_max;
	int gpuNb;

};

template<typename V, typename S> struct  TGPUplan_retro_half{
	//! Carte GPU actuelle
	char fdk;
	int device;
	//! Le volume
	V* volume_h;
	//! Le sinogramme
	S* sinogram_h;
	// Computing Architecture
	CUDABProjectionArchitecture* cudabackprojectionArchitecture;
	//! nombre d'angle calculé pour de gros sinogrammes
	int N_phi_reduit;
	//! Angle de initialisation (non utilise)
	int phi_start;
	int vn_start;
	int zn_start;
	int N_zn_par_carte;
	int N_zn_par_solverthread;
	unsigned int N_vn_par_solverthread;
	Acquisition* acquisition;
	Detector* detector;
};




#ifdef __CUDACC__
extern texture<float,cudaTextureType2DLayered> sinogram_tex0;
#endif

template<template<typename> class V, template<typename> class S,typename T> class BackProjector
{
public:
		BackProjector(Acquisition* acquisition, Detector* detector,V<T>* volume,char fdk);
	virtual ~BackProjector();
	virtual void doBackProjection(V<T>* estimatedVolume,S<T>* sinogram)=0;

	unsigned long int getProjectionNb();
	float* getAlphaIOcylinderC();
	float* getBetaIOcylinderC();
	float getGammaIOcylinderC();
	float getAlphaC();
	float getBetaC();
	Acquisition* getAcquisition() const; // Get detector
	Detector* getDetector() const; // Get detector
	
	V<T>* getVolume() const; // Get volume
	S<T>* getSinogram3D() const; // Get Sinogram

	void setAcquisition(Acquisition* acquisition); // Set detector
	void setDetector(Detector* detector); // Set detector
	
	void setVolume(V<T>* volume); // Set volume
	void setSinogram3D(S<T>* sinogram); // Set volume

	char getFdk();
	void setFdk(char fdk);

private:
	Acquisition* acquisition;
	Detector* detector;
	char fdk;
	V<T>* volume;
	S<T>* sinogram;

	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	float gammaIOcylinderC; // Lambda input-output cylinder point constant

	float alphaC;//= (M_PI*focusObjectDistance*focusDetectorDistance)/(double)(projectionNb);
	float betaC;// = (focusDetectorDistance/uDetectorPixelSize)*(focusDetectorDistance/uDetectorPixelSize);
};


template<typename V, typename S> class BackProjector_half
{
public:
	BackProjector_half(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, V* volume,char fdk);
	virtual ~BackProjector_half();
	virtual void doBackProjection(V* estimatedVolume,S* sinogram)=0;
	//virtual void doBackProjectionSFTR(V* estimatedVolume,S* sinogram)=0;//SF : trapezoid rectangle approximation
	//virtual void doBackProjectionSFTR_allCPU(V* estimatedVolume,S* sinogram)=0;//SF : trapezoid rectangle approximation
	// weighted diagonal coefficients of (HT*V*H) for SFTR matched pair
	//virtual void weightedCoeffDiagHTVHSFTR(V* coeffDiag,S* weights)=0;

	unsigned long int getProjectionNb();
	float getGammaIOcylinderC();
	Acquisition* getAcquisition() const; // Get detector
	Detector* getDetector() const; // Get detector
	CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const; // Get detector
	V* getVolume() const; // Get volume
	void setAcquisition(Acquisition* acquisition); // Set detector
	void setDetector(Detector* detector); // Set detector
	void setCUDABProjectionArchitecture(CUDABProjectionArchitecture* cudabackprojectionArchitecture); // Get detector
	void setVolume(V* volume); // Set volume
	char getFdk();
		void setFdk(char fdk);
#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif

private:
	Acquisition* acquisition;
	Detector* detector;
	CUDABProjectionArchitecture* cudabackprojectionArchitecture;
	char fdk;
	V* volume;
	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	float gammaIOcylinderC; // Lambda input-output cylinder point constant
	float alphaC;//= (M_PI*focusObjectDistance*focusDetectorDistance)/(double)(projectionNb);
	float	betaC;// = (focusDetectorDistance/uDetectorPixelSize)*(focusDetectorDistance/uDetectorPixelSize);
};

//template<template<typename> class V, template<typename> class S,typename T> class FDKBackProjector: public BackProjector<V,S,T>{
//
//public:
//	FDKBackProjector(Acquisition* acquisition, Detector* detector, V<T>* volume);
//	~FDKBackProjector();
//	float getAlphaC();
//	float getBetaC();
//
//	void doBackProjection_CPU_GPU(S<T>* sinogram, CUDABProjectionArchitecture* cudabackprojectionArchitecture);
//	void doBackProjection_GPU(S<T>* sinogram, CUDABProjectionArchitecture* cudabackprojectionArchitecture);
//	static CUT_THREADPROC solverThread(TGPUplan_retro<V, S, T> *plan);
//
//#ifdef __CUDACC__
//	__host__ void copyConstantGPU();
//#endif
//
//private:
//	float alphaC;
//	float betaC;
//};




#endif /* BACKPROJECTOR_HPP_ */
