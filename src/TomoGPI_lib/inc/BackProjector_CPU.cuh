/*
 * BackProjector_CPU.hpp
 *
 *      Author: gac
 */

#ifndef BACKPROJECTOR_CPU_HPP_
#define BACKPROJECTOR_CPU_HPP_

#include "BackProjector.cuh"
#include "Volume_CPU.cuh"
#include "Sinogram3D_CPU.cuh"

template<typename T> class VIBackProjector_compute_CUDA_mem_CPU : public BackProjector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	VIBackProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture,Volume_CPU<T>* volume,char fdk);
	~VIBackProjector_compute_CUDA_mem_CPU();
	void doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	static CUT_THREADPROC solverThread(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan);
	//void doBackProjectionDebug(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan);
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



/*template<typename T> class SFTRBackProjector_compute_CUDA_mem_CPU : public BackProjector<Volume_CPU,Sinogram3D_CPU,T>{
public:

	SFTRBackProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture,Volume_CPU<T>* volume,char fdk);
	~SFTRBackProjector_compute_CUDA_mem_CPU();
	void doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	// weighted diagonal coefficients of (HT*V*H) for SFTR matched pair
	void weightedCoeffDiagHTVHSFTR(Volume_CPU<T>* coeffDiag,Sinogram3D_CPU<T>* weights);
	static CUT_THREADPROC solverThread(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan);
	static CUT_THREADPROC solverThreadWeightedDiagonalCoefficientsSFTR(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan);
	//void doBackProjectionDebug(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan)
	//P2P
		void EnableP2P();
		void CopyConstant();
		void DisableP2P();
#ifdef __CUDACC__
__host__ void copyConstantGPU();
#endif

CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const; 
void setCUDABProjectionArchitecture(CUDABProjectionArchitecture* cudabackprojectionArchitecture); 

private:
CUDABProjectionArchitecture* cudabackprojectionArchitecture;

};*/


template<typename T> class VIBackProjector_compute_C_mem_CPU : public BackProjector<Volume_CPU,Sinogram3D_CPU,T>{
public:

	VIBackProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector, Volume_CPU<T>* volume,char fdk);
	~VIBackProjector_compute_C_mem_CPU();
	void doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//void doBackProjectionDebug(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_GPU, Sinogram3D_GPU, T> *plan);
	//P2P
		void EnableP2P();
		void CopyConstant();
		void DisableP2P();
};


/*template<typename T> class SFTRBackProjector_compute_C_mem_CPU : public BackProjector<Volume_CPU,Sinogram3D_CPU,T>{
public:

	SFTRBackProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector,Volume_CPU<T>* volume,char fdk);
	~SFTRBackProjector_compute_C_mem_CPU();
	void doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//void doBackProjectionDebug(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan);
	//P2P
		void EnableP2P();
		void CopyConstant();
		void DisableP2P();
};*/



template<typename T> class VIBackProjector_compute_OCL_mem_CPU : public BackProjector<Volume_CPU,Sinogram3D_CPU,T>{
public:

	VIBackProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector,OCLBProjectionArchitecture *oclbackprojectionArchitecture, Volume_CPU<T>* volume,char fdk);
	~VIBackProjector_compute_OCL_mem_CPU();
	void doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	void doBackProjection_FPGA(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	void doBackProjection_GPU(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	void doBackProjection_CPU(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram);
	//void doBackProjectionDebug(Volume_GPU<T>* estimatedVolume,Sinogram3D_GPU<T>* sinogram);
	//static CUT_THREADPROC solverThreadDebug(TGPUplan_retro<Volume_GPU, Sinogram3D_GPU, T> *plan);
	//P2P

		void EnableP2P();
		void CopyConstant();
		void DisableP2P();
		OCLBProjectionArchitecture* getOCLBProjectionArchitecture() const; // Get detector
void setOCLBProjectionArchitecture(OCLBProjectionArchitecture* oclbackprojectionArchitecture); // Get detector
private:
OCLBProjectionArchitecture* oclbackprojectionArchitecture;
OCLArchitecture* oclArchitecture;
};

class VIBackProjector_CPU_half : public BackProjector_half<Volume_CPU_half,Sinogram3D_CPU_half>{
public:

	VIBackProjector_CPU_half(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture,Volume_CPU_half* volume,char fdk);
	~VIBackProjector_CPU_half();
	void doBackProjection(Volume_CPU_half *estimatedVolume,Sinogram3D_CPU_half* sinogram);
	//void doBackProjectionSFTR(Volume_CPU_half *estimatedVolume,Sinogram3D_CPU_half* sinogram);//SF : trapezoid rectangle approximation
	//void doBackProjectionSFTR_allCPU(Volume_CPU_half *estimatedVolume,Sinogram3D_CPU_half* sinogram);//SF : trapezoid rectangle approximation
	// weighted diagonal coefficients of (HT*V*H) for SFTR matched pair
	//void weightedCoeffDiagHTVHSFTR(Volume_CPU_half *coeffDiag,Sinogram3D_CPU_half *weights);
	static CUT_THREADPROC solverThread(TGPUplan_retro_half<Volume_CPU_half, Sinogram3D_CPU_half> *plan);
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
