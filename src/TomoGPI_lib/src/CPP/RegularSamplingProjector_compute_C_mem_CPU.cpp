/*
 *
  * Author: gac
 */

#include "Projector_CPU.cuh"

/* RegularSamplingProjector definition */
/*template <typename T>
RegularSamplingProjector_CPU<T>::RegularSamplingProjector_CPU() : Projector<Volume_CPU,Sinogram3D_CPU,T>() {}*/

/* RegularSamplingProjector definition */
template <typename T>
RegularSamplingProjector_compute_C_mem_CPU<T>::RegularSamplingProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector, Volume_CPU<T>* volume) : Projector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector, volume){}

template <typename T>
RegularSamplingProjector_compute_C_mem_CPU<T>::~RegularSamplingProjector_compute_C_mem_CPU(){}

template <typename T>
void RegularSamplingProjector_compute_C_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume){}

template <typename T>
void RegularSamplingProjector_compute_C_mem_CPU<T>::EnableP2P(){}

template <typename T>
void RegularSamplingProjector_compute_C_mem_CPU<T>::DisableP2P(){}

#include "Projector_instances_CPU.cu"