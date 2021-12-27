/*
 * VIBackProjector_compute_C_mem_CPU.cpp
 *
  *      Author: gac
 */

#include "BackProjector_CPU.cuh"

//#include "BackProjector_MGPU.cuh"
/* VIBackProjector_CPU definition */
template<typename T>
VIBackProjector_compute_C_mem_CPU<T>::VIBackProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector,Volume_CPU<T>* volume,char fdk) : BackProjector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector, volume,fdk){}

template<typename T>
VIBackProjector_compute_C_mem_CPU<T>::~VIBackProjector_compute_C_mem_CPU(){}

template<typename T>
void VIBackProjector_compute_C_mem_CPU<T>::doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram){}

template <typename T>
void VIBackProjector_compute_C_mem_CPU<T>::EnableP2P(){}

template <typename T>
void VIBackProjector_compute_C_mem_CPU<T>::DisableP2P(){}

#include "BackProjector_instances_CPU.cu"