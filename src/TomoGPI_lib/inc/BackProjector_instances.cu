template class BackProjector<Volume_CPU, Sinogram3D_CPU,float>; // 32-bit unsigned image
template class BackProjector<Volume_CPU, Sinogram3D_CPU,double>; // 64-bit signed image

template class BackProjector<Volume_GPU, Sinogram3D_GPU,float>; // 32-bit unsigned image
template class BackProjector<Volume_GPU, Sinogram3D_GPU,double>; // 64-bit signed image

template class BackProjector_half<Volume_CPU_half, Sinogram3D_CPU_half>; // 32-bit unsigned image
