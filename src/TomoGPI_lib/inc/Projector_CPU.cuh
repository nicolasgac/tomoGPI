/*
 * Projector_CPU.cuh
  *
 *      Author: gac
 */

#ifndef PROJECTOR_CPU_HPP_
#define PROJECTOR_CPU_HPP_

#include "Projector.cuh"
#include "Volume_CPU.cuh"
#include "Sinogram3D_CPU.cuh"

template<typename T> class RegularSamplingProjector_compute_C_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	RegularSamplingProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector,Volume_CPU<T>* volume);
	~RegularSamplingProjector_compute_C_mem_CPU();
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();
};

/*template<typename T> class SFTRProjector_compute_C_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	SFTRProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector,Volume_CPU<T>* volume);
	~SFTRProjector_compute_C_mem_CPU();
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();
};*/

template<typename T> class SiddonProjector_compute_C_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	SiddonProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector,Volume_CPU<T>* volume);
	~SiddonProjector_compute_C_mem_CPU();
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();
};

template<typename T> class SiddonProjector_compute_OCL_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	SiddonProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector, OCLProjectionArchitecture *oclprojectionArchitecture, Volume_CPU<T>* volume);
	~SiddonProjector_compute_OCL_mem_CPU();
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();

	OCLProjectionArchitecture* getOCLProjectionArchitecture() const; // Get detector
	void setOCLProjectionArchitecture(OCLProjectionArchitecture* oclprojectionArchitecture); // Get detector
	private:
	OCLProjectionArchitecture* oclprojectionArchitecture;
	OCLArchitecture* oclArchitecture;
};

template<typename T> class RegularSamplingProjector_compute_OCL_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	RegularSamplingProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector,OCLProjectionArchitecture *oclprojectionArchitecture,Volume_CPU<T>* volume);
	~RegularSamplingProjector_compute_OCL_mem_CPU();

	static CUT_THREADPROC solverThread(TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T > *plan);
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//static void projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T> *plan,T* sinogram_d,Sinogram3D_CPU<T> *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams);
	//void doProjectionDebug(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();

	OCLProjectionArchitecture* getOCLProjectionArchitecture() const; // Get detector
	void setOCLProjectionArchitecture(OCLProjectionArchitecture* oclprojectionArchitecture); // Get detector
	private:
	OCLProjectionArchitecture* oclprojectionArchitecture;
	OCLArchitecture* oclArchitecture;
};

template<typename T> class RegularSamplingProjector_compute_CUDA_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	RegularSamplingProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,Volume_CPU<T>* volume);
	~RegularSamplingProjector_compute_CUDA_mem_CPU();
	static CUT_THREADPROC solverThread(TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T > *plan);
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//static void projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T> *plan,T* sinogram_d,Sinogram3D_CPU<T> *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams);
	//void doProjectionDebug(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const; // Get detector
	void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set detector
	//P2P
	void EnableP2P();
	void DisableP2P();
#ifdef __CUDACC__
	__host__ void  copyConstantGPU();
#endif	
private:
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
};

/*template<typename T> class SFTRProjector_compute_CUDA_mem_CPU : public Projector<Volume_CPU,Sinogram3D_CPU,T>{
public:
	SFTRProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,Volume_CPU<T>* volume);
	~SFTRProjector_compute_CUDA_mem_CPU();

	static CUT_THREADPROC solverThread(TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T > *plan);
	static CUT_THREADPROC solverThreadWeightedDiagonalCoefficientsSFTR(TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T > *plan);
	void doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	// all on CPU
	void doProjection_v0(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);//SFTR : 2 kernels
	void doProjection_v2(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);//separable footprint : rectangle rectangle approximation
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR(Sinogram3D_CPU<T>* coeffDiag,Volume_CPU<T>* weights);
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_CPU<T>* coeffDiag,Volume_CPU<T>* weights);
	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const; // Get detector
	void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set detector
	//static void projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj<Volume_CPU,Sinogram3D_CPU, T> *plan,T* sinogram_d,Sinogram3D_CPU<T> *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams);
	//void doProjectionDebug(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>*volume);
	//P2P
	void EnableP2P();
	void DisableP2P();
#ifdef __CUDACC__
	__host__ void  copyConstantGPU();
#endif	
private:
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
};*/


class RegularSamplingProjector_CPU_half : public Projector_half<Volume_CPU_half,Sinogram3D_CPU_half>{
public:
	RegularSamplingProjector_CPU_half(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,Volume_CPU_half* volume);
	RegularSamplingProjector_CPU_half();
	~RegularSamplingProjector_CPU_half();

	static CUT_THREADPROC solverThread(TGPUplan_proj_half<Volume_CPU_half,Sinogram3D_CPU_half> *plan);

	void doProjection(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume);
	void doProjectionSFTR(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume);
	void doProjectionSFTR_2kernels(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half* volume);
	void doProjectionSFTR_opti(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume);
	void doProjectionSFTR_allCPU(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume);

	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR(Sinogram3D_CPU_half* coeffDiag,Volume_CPU_half* weights);
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_CPU_half* coeffDiag,Volume_CPU_half* weights);

	static void projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj_half<Volume_CPU_half,Sinogram3D_CPU_half> *plan,half* sinogram_d,Sinogram3D_CPU_half *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams);

	//#ifdef __CUDACC__
	//	__host__ void  copyConstantGPU();
	//#endif

	//private:
	//	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	//	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	//	float gammaIOcylinderC; // Lambda input-output cylinder point constant
};


#endif /* PROJECTOR_HPP_ */
