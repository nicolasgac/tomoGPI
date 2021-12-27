/*
 * Mex.cuh
 *
 *      Author: gac
 */

#ifndef MEX_HPP_
#define MEX_HPP_

#include "mex.h"
#include "class_handle_iter.hpp"
#include <omp.h>
#include <math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <multithreading.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


#include "ConfigTiff.hpp"
#include "ConfigCT.hpp"
#include "ConfigComputeArchitecture.hpp"
#include "ConfigIteration.hpp"
#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"
#include "Image3D.cuh"
#include "Image3D_CPU.cuh"
#include "Image3D_GPU.cuh"
//#include "Image3D_MGPU.cuh"
#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Projector.cuh"
#include "Projector_CPU.cuh"
#include "Projector_GPU.cuh"
//#include "Projector_MGPU.cuh"
#include "BackProjector.cuh"
#include "BackProjector_CPU.cuh"
#include "BackProjector_GPU.cuh"
//#include "BackProjector_MGPU.cuh"
#include "ComputingArchitecture.cuh"
#include "Convolution3D.cuh"
#include "Convolution3D_CPU.cuh"
#include "Convolution3D_GPU.cuh"
//#include "Convolution3D_MGPU.cuh"
#include "HuberRegularizer_CPU.cuh"
#include "HuberRegularizer_GPU.cuh"
//#include "HuberRegularizer_MGPU.cuh"
#include "GeneralizedGaussianRegularizer_CPU.cuh"
#include "GeneralizedGaussianRegularizer_GPU.cuh"
//#include "GeneralizedGaussianRegularizer_MGPU.cuh"
#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"


/*typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_CPU;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_GPU;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_CPU_half;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_GPU_half;*/

template<template<typename> class I, template<typename> class V,template<typename> class S,template<typename> class C,typename T> class Mex{

public:

	Mex(string workdirectory);

	virtual ~Mex();
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;
	I<T>* getIter3D() const;
		void setIter3D(I<T>*  iter) ;

	void doMexIter( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[],char *cmd);
private:

I<T> *iter3D;

};

template<typename T> class Mex_CPU : public Mex<Iter3D_RSVI_compute_CUDA_mem_CPU,Volume_CPU,Sinogram3D_CPU,Convolution3D_CPU,T>{
	
public:
	Mex_CPU(string workdirectory);
	~Mex_CPU();

};


template<typename T> class Mex_GPU : public Mex<Iter3D_RSVI_compute_CUDA_mem_GPU,Volume_GPU,Sinogram3D_GPU,Convolution3D_GPU,T>{

public:
	Mex_GPU(string workdirectory);
	~Mex_GPU();

};


template<typename I,typename V,typename S,typename C> class Mex_half{

public:

	Mex_half();
	virtual ~Mex_half();
	//virtual void doProjection(S* estimatedSinogram) = 0;

	int doMexIter( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
private:

};

class Mex_CPU_half : public Mex_half<Iter3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half,Convolution3D_CPU_half>{

public:
	Mex_CPU_half(string workdirectory);
	~Mex_CPU_half();


};

class Mex_GPU_half : public Mex_half<Iter3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half,Convolution3D_GPU_half>{

public:
	Mex_GPU_half(string workdirectory);
	~Mex_GPU_half();

	//void doProjection(Sinogram3D_GPU_half* estimatedSinogram);

	//#ifdef __CUDACC__
	//	__host__ void  copyConstantGPU();
	//#endif

	//private:
	//	float *alphaIOcylinderC; // Alpha input-output cylinder point constant tab
	//	float *betaIOcylinderC; // Beta input-output cylinder point constant tab
	//	float gammaIOcylinderC; // Lambda input-output cylinder point constant
};



#endif /* MEX_HPP_ */
