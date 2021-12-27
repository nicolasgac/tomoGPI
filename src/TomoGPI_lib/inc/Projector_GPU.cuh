/*
 * Projector_GPU.cuh
 *
 *      Author: gac
 */

#ifndef PROJECTOR_GPU_HPP_
#define PROJECTOR_GPU_HPP_

#include "Projector.cuh"
#include "Volume_GPU.cuh"
#include "Sinogram3D_GPU.cuh"


template<typename T> class RegularSamplingProjector_compute_CUDA_mem_GPU : public Projector<Volume_GPU,Sinogram3D_GPU,T>{
public:
	RegularSamplingProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,Volume_GPU<T>* volume);
	//RegularSamplingProjector_GPU();

	virtual ~RegularSamplingProjector_compute_CUDA_mem_GPU();

	void doProjection(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>* volume);
	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const; // Get detector
	void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set detector
	//void doProjectionDebug(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>* volume);

	//P2P
	void EnableP2P();
	void DisableP2P();

#ifdef __CUDACC__
	__host__ void  copyConstantGPU();
#endif
private:
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
};


/*template<typename T> class SFTRProjector_compute_CUDA_mem_GPU : public Projector<Volume_GPU,Sinogram3D_GPU,T>{
public:
	SFTRProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture,Volume_GPU<T>* volume);
	virtual ~SFTRProjector_compute_CUDA_mem_GPU();

	void doProjection(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>* volume);
	// use GPU
	void doProjection_v0(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>*volume);//SFTR : 2 kernels
	void doProjection_v2(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>*volume);//separable footprint : trapezoid rectangle approximation
	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const; // Get detector
	void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set detector
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR(Sinogram3D_GPU<T>* coeffDiag,Volume_GPU<T>* weights);
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_GPU<T>* coeffDiag,Volume_GPU<T>* weights);
	
	//void doProjectionDebug(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T>* volume);

	//P2P
	void EnableP2P();
	void DisableP2P();

#ifdef __CUDACC__
	__host__ void  copyConstantGPU();
#endif
private:
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
};*/



class RegularSamplingProjector_GPU_half : public Projector_half<Volume_GPU_half,Sinogram3D_GPU_half>{
public:
	RegularSamplingProjector_GPU_half();
	RegularSamplingProjector_GPU_half(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume_GPU_half* volume);
	~RegularSamplingProjector_GPU_half();

	void doProjection(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume);
	void doProjectionSFTR(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume);
	void doProjectionSFTR_2kernels(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half* volume);
	void doProjectionSFTR_opti(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume);
	void doProjectionSFTR_allCPU(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume);

	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR(Sinogram3D_GPU_half* coeffDiag,Volume_GPU_half* weights);
	// weighted diagonal coefficients of (H*V*HT) for SFTR matched pair
	void weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_GPU_half* coeffDiag,Volume_GPU_half* weights);

	//#ifdef __CUDACC__
	//	__host__ void  copyConstantGPU();
	//#endif

	//private:
	//	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	//	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	//	float gammaIOcylinderC; // Lambda input-output cylinder point constant
};


#endif /* PROJECTOR_HPP_ */
