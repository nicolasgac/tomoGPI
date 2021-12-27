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
RegularSamplingProjector_compute_OCL_mem_CPU<T>::RegularSamplingProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector, OCLProjectionArchitecture *oclprojectionArchitecture,Volume_CPU<T>* volume) : Projector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector,volume){
	this->setOCLProjectionArchitecture(oclprojectionArchitecture);
}

template <typename T>
RegularSamplingProjector_compute_OCL_mem_CPU<T>::~RegularSamplingProjector_compute_OCL_mem_CPU(){}

template <typename T>
void RegularSamplingProjector_compute_OCL_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume)
{
	//std::cout << "\tRegular Sampling Projection running on CPU " << sched_getcpu() << std::endl;
	//std::cout << "\tAdress sinogram " << estimatedSinogram->getDataSinogram() << std::endl;

	std::cout << "\tOCL Regular Sampling Projection" << std::endl;
}

template <typename T>
void RegularSamplingProjector_compute_OCL_mem_CPU<T>::EnableP2P(){}

template <typename T>
void RegularSamplingProjector_compute_OCL_mem_CPU<T>::DisableP2P(){}


template<typename T>
OCLProjectionArchitecture* RegularSamplingProjector_compute_OCL_mem_CPU<T>::getOCLProjectionArchitecture() const
{
	return this->oclprojectionArchitecture;
}

template<typename T>
void RegularSamplingProjector_compute_OCL_mem_CPU<T>::setOCLProjectionArchitecture(OCLProjectionArchitecture*  oclprojectionArchitecture)
{
	this->oclprojectionArchitecture =  oclprojectionArchitecture;
}
#include "Projector_instances_CPU.cu"