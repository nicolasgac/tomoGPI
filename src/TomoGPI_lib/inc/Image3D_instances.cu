#include "half_float_conversion_kernel.cuh"
template class Image3D<unsigned char>; // 16-bit signed image
template class Image3D<half>; // 16-bit signed image
template class Image3D<char>; // 16-bit signed image
template class Image3D<unsigned short>; // 16-bit signed image
template class Image3D<short>; // 16-bit signed image
template class Image3D<int>; // 32-bit signed image
template class Image3D<float>; // 32-bit real image
template class Image3D<double>; // 64-bit real image
template class Image3D<bool>; // image of booleans