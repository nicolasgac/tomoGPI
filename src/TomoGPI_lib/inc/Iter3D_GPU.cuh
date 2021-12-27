/*
 * iter3d_GPU.hpp
 *
 *      Author: gac
 */

#ifndef ITER3D_GPU_HPP_
#define ITER3D_GPU_HPP_

#include "Iter3D.cuh"

template<typename T> class Iter3D_RSVI_compute_CUDA_mem_GPU : public Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_GPU,VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>{
public:
	Iter3D_RSVI_compute_CUDA_mem_GPU(string workdirectory);
	Iter3D_RSVI_compute_CUDA_mem_GPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_RSVI_compute_CUDA_mem_GPU();
};

/*template<typename T> class Iter3D_SFTR_compute_CUDA_mem_GPU : public Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_GPU,SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>{
public:
	Iter3D_SFTR_compute_CUDA_mem_GPU(string workdirectory);
	Iter3D_SFTR_compute_CUDA_mem_GPU(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_SFTR_compute_CUDA_mem_GPU();
};*/

class Iter3D_GPU_half : public Iter3D_half<RegularSamplingProjector_GPU_half,VIBackProjector_GPU_half,HuberRegularizer_GPU_half,GeneralizedGaussianRegularizer_GPU_half,Convolution3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half>{
public:
	Iter3D_GPU_half(string workdirectory);
	Iter3D_GPU_half(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	~Iter3D_GPU_half();
};


#endif /* ITER3D_HPP_ */
