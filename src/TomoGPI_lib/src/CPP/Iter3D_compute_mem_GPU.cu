/*
 * Iter3D_compute_mem_GPU.cu
 *
 *      Author: gac
 */

#include "Iter3D_GPU.cuh"

template <typename T>
Iter3D_RSVI_compute_CUDA_mem_GPU<T>::Iter3D_RSVI_compute_CUDA_mem_GPU(string workdirectory) : Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_GPU,VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>(workdirectory){}

template <typename T>
Iter3D_RSVI_compute_CUDA_mem_GPU<T>::Iter3D_RSVI_compute_CUDA_mem_GPU(string workdirectory,ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_GPU,VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>(workdirectory,configComputeArchitecture_file){}

template <typename T>
Iter3D_RSVI_compute_CUDA_mem_GPU<T>::~Iter3D_RSVI_compute_CUDA_mem_GPU(){}

/*
template <typename T>
Iter3D_SFTR_compute_CUDA_mem_GPU<T>::Iter3D_SFTR_compute_CUDA_mem_GPU(string workdirectory) : Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_GPU,SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>(workdirectory){}

template <typename T>
Iter3D_SFTR_compute_CUDA_mem_GPU<T>::Iter3D_SFTR_compute_CUDA_mem_GPU(string workdirectory,ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_GPU,SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,T>(workdirectory,configComputeArchitecture_file){}

template <typename T>
Iter3D_SFTR_compute_CUDA_mem_GPU<T>::~Iter3D_SFTR_compute_CUDA_mem_GPU(){}
*/

#include "Iter3D_instances_GPU.cu"
