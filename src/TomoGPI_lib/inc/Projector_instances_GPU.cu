
template class Projector<Volume_GPU, Sinogram3D_GPU,float>; // 32-bit unsigned image
template class Projector<Volume_GPU, Sinogram3D_GPU,double>; // 64-bit signed image

template class Projector_half<Volume_GPU_half, Sinogram3D_GPU_half>; // 8-bit float image

template class RegularSamplingProjector_compute_CUDA_mem_GPU<float>; // 32-bit unsigned image
template class RegularSamplingProjector_compute_CUDA_mem_GPU<double>; // 64-bit signed image
