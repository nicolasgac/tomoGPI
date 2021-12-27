/*
 * BackProjector_GPU.hpp
 *
 *      Author: gac
  */

#ifndef BACKPROJECTOR_GPU_HPP_
#define BACKPROJECTOR_GPU_HPP_

#include "BackProjector.cuh"
#include "Volume_GPU.cuh"
#include "Sinogram3D_GPU.cuh"


template<typename T> class VIBackProjector_compute_CUDA_mem_GPU : public BackProjector<Volume_GPU,Sinogram3D_GPU,T>{
public:

	VIBackProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume_GPU<T>* volume,char fdk);
	~VIBackProjector_compute_CUDA_mem_GPU();

	void doBackProjection(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	//void doBackProjectionDebug(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_GPU, Sinogram3D_GPU, T> *plan);
	//P2P
		void EnableP2P();
		void CopyConstant();
		void DisableP2P();

#ifdef __CUDACC__
__host__ void copyConstantGPU();
#endif

CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const; // Get detector
void setCUDABProjectionArchitecture(CUDABProjectionArchitecture* cudabackprojectionArchitecture); // Get detector
private:
CUDABProjectionArchitecture* cudabackprojectionArchitecture;

};



/*template<typename T> class SFTRBackProjector_compute_CUDA_mem_GPU : public BackProjector<Volume_GPU,Sinogram3D_GPU,T>{
public:

	SFTRBackProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume_GPU<T>* volume,char fdk);
	~SFTRBackProjector_compute_CUDA_mem_GPU();

	void doBackProjection(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	// weighted diagonal coefficients of (HT*V*H) for SFTR matched pair
	void weightedCoeffDiagHTVHSFTR(Volume_GPU<T>* coeffDiag,Sinogram3D_GPU<T>* weights);
	//void doBackProjectionDebug(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_GPU, Sinogram3D_GPU, T> *plan);
	//P2P
		void EnableP2P();
		void CopyConstant();
		void DisableP2P();

#ifdef __CUDACC__
__host__ void copyConstantGPU();
#endif
CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const; // Get detector
void setCUDABProjectionArchitecture(CUDABProjectionArchitecture* cudabackprojectionArchitecture); // Get detector
private:
CUDABProjectionArchitecture* cudabackprojectionArchitecture;
};*/


class VIBackProjector_GPU_half : public BackProjector_half<Volume_GPU_half,Sinogram3D_GPU_half>{
public:

	VIBackProjector_GPU_half(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume_GPU_half* volume,char fdk);
	~VIBackProjector_GPU_half();
	void doBackProjection(Volume_GPU_half *estimatedVolume,Sinogram3D_GPU_half* sinogram);
	//void doBackProjectionSFTR(Volume_GPU_half *estimatedVolume,Sinogram3D_GPU_half* sinogram);//SF : trapezoid rectangle approximation
	//void doBackProjectionSFTR_allCPU(Volume_GPU_half *estimatedVolume,Sinogram3D_GPU_half* sinogram);//SF : trapezoid rectangle approximation
	// weighted diagonal coefficients of (HT*V*H) for SFTR matched pair
	//void weightedCoeffDiagHTVHSFTR(Volume_GPU_half* coeffDiag,Sinogram3D_GPU_half* weights);


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
