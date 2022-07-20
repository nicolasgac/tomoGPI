/*
 * iter3d_CPU.hpp
 *
 *      Author: gac
 */

#ifndef ITER3D_CPU_HPP_
#define ITER3D_CPU_HPP_

#include "Iter3D.cuh"

template<typename T> class Iter3D_RSVI_compute_C_mem_CPU : public Iter3D_compute_C<RegularSamplingProjector_compute_C_mem_CPU,VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_RSVI_compute_C_mem_CPU(string workdirectory);
	Iter3D_RSVI_compute_C_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_RSVI_compute_C_mem_CPU();
};

template<typename T> class Iter3D_RSVI_compute_OCL_mem_CPU : public Iter3D_compute_OCL<RegularSamplingProjector_compute_OCL_mem_CPU,VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_RSVI_compute_OCL_mem_CPU(string workdirectory);
	Iter3D_RSVI_compute_OCL_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_RSVI_compute_OCL_mem_CPU();
};

template<typename T> class Iter3D_RSVI_compute_CUDA_mem_CPU : public Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_CPU,VIBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_RSVI_compute_CUDA_mem_CPU(string workdirectory);
	Iter3D_RSVI_compute_CUDA_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_RSVI_compute_CUDA_mem_CPU();
};


template<typename T> class Iter3D_RSVI_compute_CUDA_OCL_mem_CPU : public Iter3D_compute_CUDA_OCL<RegularSamplingProjector_compute_CUDA_mem_CPU,VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_RSVI_compute_CUDA_OCL_mem_CPU(string workdirectory);
	Iter3D_RSVI_compute_CUDA_OCL_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_RSVI_compute_CUDA_OCL_mem_CPU();
};


/*template<typename T> class Iter3D_SFTR_compute_C_mem_CPU : public Iter3D_compute_C<SFTRProjector_compute_C_mem_CPU,SFTRBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_SFTR_compute_C_mem_CPU(string workdirectory);
	Iter3D_SFTR_compute_C_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_SFTR_compute_C_mem_CPU();
};*/

/*template<typename T> class Iter3D_SFTR_compute_CUDA_mem_CPU : public Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_CPU,SFTRBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_SFTR_compute_CUDA_mem_CPU(string workdirectory);
	Iter3D_SFTR_compute_CUDA_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_SFTR_compute_CUDA_mem_CPU();
};*/

template<typename T> class Iter3D_SiddonVI_compute_C_mem_CPU : public Iter3D_compute_C<SiddonProjector_compute_C_mem_CPU,VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_SiddonVI_compute_C_mem_CPU(string workdirectory);
	Iter3D_SiddonVI_compute_C_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_SiddonVI_compute_C_mem_CPU();
};

template<typename T> class Iter3D_SiddonVI_compute_OCL_mem_CPU : public Iter3D_compute_OCL<SiddonProjector_compute_OCL_mem_CPU,VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,T>{
public:
	Iter3D_SiddonVI_compute_OCL_mem_CPU(string workdirectory);
	Iter3D_SiddonVI_compute_OCL_mem_CPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_SiddonVI_compute_OCL_mem_CPU();
};

class Iter3D_CPU_half : public Iter3D_half<RegularSamplingProjector_CPU_half,VIBackProjector_CPU_half,HuberRegularizer_CPU_half,GeneralizedGaussianRegularizer_CPU_half,Convolution3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half>{
public:
	Iter3D_CPU_half(string workdirectory);
	Iter3D_CPU_half(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_CPU_half();
};


#endif /* ITER3D_HPP_ */
