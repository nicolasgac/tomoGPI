/*
 * GPUConstant.cuh
 *
 *      Author: gac
 */

#ifndef GPUCONSTANT_CUH_
#define GPUCONSTANT_CUH_

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define MAX_PROJECTION 2400

#define MAX_SIZE_H_KERNEL 21
#define MAX_SIZE_V_KERNEL 21
#define MAX_SIZE_P_KERNEL 21
// Nicolas

#define MAX_CLASSE 20

/* Regular Sampling projector constant */
extern __device__ __constant__ float alphaIOcylinderC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float betaIOcylinderC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float gammaIOcylinderC_GPU;
extern __device__ __constant__ float xVolumeCenterPixel_GPU;
extern __device__ __constant__ float yVolumeCenterPixel_GPU;
extern __device__ __constant__ float zVolumeCenterPixel_GPU;
extern __device__ __constant__ float xVolumePixelSize_GPU;
extern __device__ __constant__ unsigned long int xVolumePixelNb_GPU;
extern __device__ __constant__ unsigned long int yVolumePixelNb_GPU;
extern __device__ __constant__ unsigned long int zVolumePixelNb_GPU;

/* Joseph projector constant */
#ifdef JOSEPH
extern __device__ __constant__ float alphaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float betaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float deltaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float sigmaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float kappaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float iotaPreComputingC_GPU[MAX_PROJECTION];
extern __device__ __constant__ float gammaPrecomputingC_GPU;
extern __device__ __constant__ float omegaPrecomputingC_GPU;
#endif

/* FDK BackProjection constant*/
extern __device__ __constant__ float alphaC_GPU;
extern __device__ __constant__ float betaC_GPU;

/* Convolution constant */
extern __device__ __constant__  float c_Kernel_h[MAX_SIZE_H_KERNEL];
extern __device__ __constant__  float c_Kernel_v[MAX_SIZE_V_KERNEL];
extern __device__ __constant__  float c_Kernel_p[MAX_SIZE_P_KERNEL];
extern __device__ __constant__  unsigned long int c_volume_x;
extern __device__ __constant__  unsigned long int c_volume_y;
extern __device__ __constant__  unsigned long int c_volume_z;
extern __device__ __constant__  unsigned long int c_kernel_radius_x;
extern __device__ __constant__  unsigned long int c_kernel_radius_y;
extern __device__ __constant__  unsigned long int c_kernel_radius_z;

/* Acquisition constant */
extern __device__ __constant__ float focusDetectorDistance_GPU;
extern __device__ __constant__ float focusObjectDistance_GPU;

/* Detector constant */
extern __device__ __constant__ float uDetectorCenterPixel_GPU;
extern __device__ __constant__ float vDetectorCenterPixel_GPU;
extern __device__ __constant__ float uDetectorPixelSize_GPU;
extern __device__ __constant__ float vDetectorPixelSize_GPU;
extern __device__ __constant__ unsigned long int uDetectorPixelNb_GPU;
extern __device__ __constant__ unsigned long int vDetectorPixelNb_GPU;
extern __device__ __constant__ unsigned long int projectionNb_GPU;

/* Sinogram_GPU constant */
extern __device__ __constant__ unsigned long int uSinogramPixelNb_GPU;
extern __device__ __constant__ unsigned long int vSinogramPixelNb_GPU;

/* Segmentation GPU constant */
//Nicolas
extern __device__ __constant__ int Kclasse;
extern __device__ __constant__ double gammaPotts;//Potts coefficient
extern __device__ __constant__ double meanclasses[MAX_CLASSE];//means of the classes
extern __device__ __constant__ double varianceclasses[MAX_CLASSE];//variances of the classes
extern __device__ __constant__ double energySingleton[MAX_CLASSE];//energies of singleton
#endif 
//__host__ void RegularSamplingProjector_CPU_half::copyConstantGPU()
//{
//	unsigned long int projectionNb = (this->getAcquisition())->getProjectionNb();
//
//	Projector_half<Volume_CPU_half,Sinogram3D_CPU_half>::copyConstantGPU();
//
//	cudaMemcpyToSymbol(alphaIOcylinderC_GPU,this->alphaIOcylinderC,projectionNb*sizeof(float));
//	cudaMemcpyToSymbol(betaIOcylinderC_GPU,this->betaIOcylinderC,projectionNb*sizeof(float));
//	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&this->gammaIOcylinderC,sizeof(float));
//}
//
//__host__ void RegularSamplingProjector_GPU_half::copyConstantGPU()
//{
//	unsigned long int projectionNb = (this->getAcquisition())->getProjectionNb();
//
//	Projector_half<Volume_GPU_half,Sinogram3D_GPU_half>::copyConstantGPU();
//
//	cudaMemcpyToSymbol(alphaIOcylinderC_GPU,this->alphaIOcylinderC,projectionNb*sizeof(float));
//	cudaMemcpyToSymbol(betaIOcylinderC_GPU,this->betaIOcylinderC,projectionNb*sizeof(float));
//	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&this->gammaIOcylinderC,sizeof(float));
//}

//template <template<typename> class V,template<typename> class S, typename T>
//__host__ void JosephProjector<V,S,T>::copyConstantGPU()
//{
//	//	unsigned long int projectionNb = (this->getAcquisition())->getProjectionNb();
//	//	cudaMemcpyToSymbol(alphaPreComputingC_GPU,this->alphaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(betaPreComputingC_GPU,this->betaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(deltaPreComputingC_GPU,this->deltaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(sigmaPreComputingC_GPU,this->sigmaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(kappaPreComputingC_GPU,this->kappaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(iotaPreComputingC_GPU,this->iotaPreComputingC,projectionNb*sizeof(float));
//	//	cudaMemcpyToSymbol(gammaPrecomputingC_GPU,&this->gammaPrecomputingC,sizeof(float));
//	//	cudaMemcpyToSymbol(omegaPrecomputingC_GPU,&this->omegaPrecomputingC,sizeof(float));
//}

#endif /* GPUCONSTANT_CUH_ */
