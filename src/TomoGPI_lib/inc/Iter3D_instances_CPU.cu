template class Iter3D<RegularSamplingProjector_compute_OCL_mem_CPU, VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<RegularSamplingProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<RegularSamplingProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
template class Iter3D<RegularSamplingProjector_compute_CUDA_mem_CPU, VIBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<RegularSamplingProjector_compute_CUDA_mem_CPU, VIBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image

/*template class Iter3D<SFTRProjector_compute_C_mem_CPU, SFTRBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<SFTRProjector_compute_C_mem_CPU, SFTRBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
template class Iter3D<SFTRProjector_compute_CUDA_mem_CPU, SFTRBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<SFTRProjector_compute_CUDA_mem_CPU, SFTRBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
*/

template class Iter3D<SiddonProjector_compute_OCL_mem_CPU, VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<SiddonProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D<SiddonProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image

template class Iter3D_compute_OCL<RegularSamplingProjector_compute_OCL_mem_CPU, VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_C<RegularSamplingProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_C<RegularSamplingProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
template class Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_CPU, VIBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_CUDA<RegularSamplingProjector_compute_CUDA_mem_CPU, VIBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image

/*template class Iter3D_compute_C<SFTRProjector_compute_C_mem_CPU, SFTRBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_C<SFTRProjector_compute_C_mem_CPU, SFTRBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
template class Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_CPU, SFTRBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_CUDA<SFTRProjector_compute_CUDA_mem_CPU, SFTRBackProjector_compute_CUDA_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image
*/

template class Iter3D_compute_OCL<SiddonProjector_compute_OCL_mem_CPU, VIBackProjector_compute_OCL_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_C<SiddonProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Iter3D_compute_C<SiddonProjector_compute_C_mem_CPU, VIBackProjector_compute_C_mem_CPU,HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU,Convolution3D_CPU,Volume_CPU,Sinogram3D_CPU,double>; // 64-bit signed image

template class Iter3D_RSVI_compute_OCL_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_RSVI_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_RSVI_compute_C_mem_CPU<double>; // 64-bit signed imag
template class Iter3D_RSVI_compute_CUDA_OCL_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_RSVI_compute_CUDA_OCL_mem_CPU<double>; // 64-bit signed imag
template class Iter3D_RSVI_compute_CUDA_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_RSVI_compute_CUDA_mem_CPU<double>; // 64-bit signed imag

template class Iter3D_SiddonVI_compute_OCL_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_SiddonVI_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_SiddonVI_compute_C_mem_CPU<double>; // 64-bit signed imag

/*template class Iter3D_SFTR_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_SFTR_compute_C_mem_CPU<double>; // 64-bit signed image
template class Iter3D_SFTR_compute_CUDA_mem_CPU<float>; // 32-bit unsigned image
template class Iter3D_SFTR_compute_CUDA_mem_CPU<double>; // 64-bit signed image
*/

template class Iter3D_half<RegularSamplingProjector_CPU_half, VIBackProjector_CPU_half,HuberRegularizer_CPU_half,GeneralizedGaussianRegularizer_CPU_half,Convolution3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half>; // 8-bit float image