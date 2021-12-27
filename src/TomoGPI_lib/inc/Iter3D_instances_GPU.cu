template class Iter3D<RegularSamplingProjector_compute_CUDA_mem_GPU, VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,float>; // 32-bit unsigned image
template class Iter3D<RegularSamplingProjector_compute_CUDA_mem_GPU, VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,double>; // 64-bit signed image

/*template class Iter3D<SFTRProjector_compute_CUDA_mem_GPU, SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,float>; // 32-bit unsigned image
template class Iter3D<SFTRProjector_compute_CUDA_mem_GPU, SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,double>; // 64-bit signed image
*/


template class Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_GPU, VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,float>; // 32-bit unsigned image
template class Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_GPU, VIBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,double>; // 64-bit signed image

/*template class Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_GPU, SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,float>; // 32-bit unsigned image
template class Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_GPU, SFTRBackProjector_compute_CUDA_mem_GPU,HuberRegularizer_GPU,GeneralizedGaussianRegularizer_GPU,Convolution3D_GPU,Volume_GPU,Sinogram3D_GPU,double>; // 64-bit signed image
*/

template class Iter3D_RSVI_compute_CUDA_mem_GPU<float>; // 32-bit unsigned image
template class Iter3D_RSVI_compute_CUDA_mem_GPU<double>; // 64-bit signed image

//template class Iter3D_SFTR_compute_CUDA_mem_GPU<float>; // 32-bit unsigned image
//template class Iter3D_SFTR_compute_CUDA_mem_GPU<double>; // 64-bit signed imag

template class Iter3D_half<RegularSamplingProjector_GPU_half, VIBackProjector_GPU_half,HuberRegularizer_GPU_half,GeneralizedGaussianRegularizer_GPU_half,Convolution3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half>; // 8-bit float image
