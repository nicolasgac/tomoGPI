
template class Projector<Volume_CPU, Sinogram3D_CPU,int>; // 16-bit signed image
template class Projector<Volume_CPU, Sinogram3D_CPU,short>; // 16-bit signed image
template class Projector<Volume_CPU, Sinogram3D_CPU,float>; // 32-bit unsigned image
template class Projector<Volume_CPU, Sinogram3D_CPU,double>; // 64-bit signed image


template class Projector_half<Volume_CPU_half, Sinogram3D_CPU_half>; // 8-bit float image


template class RegularSamplingProjector_compute_C_mem_CPU<short>; // 32-bit unsigned image
template class RegularSamplingProjector_compute_C_mem_CPU<int>; // 64-bit signed image
template class RegularSamplingProjector_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class RegularSamplingProjector_compute_C_mem_CPU<double>; // 64-bit signed image

template class RegularSamplingProjector_compute_CUDA_mem_CPU<short>; // 32-bit unsigned image
template class RegularSamplingProjector_compute_CUDA_mem_CPU<int>; // 64-bit signed image
template class RegularSamplingProjector_compute_CUDA_mem_CPU<float>; // 32-bit unsigned image
template class RegularSamplingProjector_compute_CUDA_mem_CPU<double>; // 64-bit signed image

template class RegularSamplingProjector_compute_OCL_mem_CPU<float>; // 32-bit unsigned image


/*template class SFTRProjector_compute_C_mem_CPU<short>; // 32-bit unsigned image
template class SFTRProjector_compute_C_mem_CPU<int>; // 64-bit signed image
template class SFTRProjector_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class SFTRProjector_compute_C_mem_CPU<double>; // 64-bit signed image

template class SFTRProjector_compute_CUDA_mem_CPU<short>; // 32-bit unsigned image
template class SFTRProjector_compute_CUDA_mem_CPU<int>; // 64-bit signed image
template class SFTRProjector_compute_CUDA_mem_CPU<float>; // 32-bit unsigned image
template class SFTRProjector_compute_CUDA_mem_CPU<double>; // 64-bit signed image
*/

template class SiddonProjector_compute_C_mem_CPU<float>; // 32-bit unsigned image
template class SiddonProjector_compute_C_mem_CPU<double>; // 64-bit signed image

template class SiddonProjector_compute_OCL_mem_CPU<float>; // 32-bit unsigned image