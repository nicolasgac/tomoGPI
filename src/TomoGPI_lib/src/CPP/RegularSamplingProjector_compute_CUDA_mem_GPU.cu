/*
 *
  * Author: gac
 */

#include "Projector_GPU.cuh"
#include "Projector_kernel.cuh"



/*#include "projection_sftr_mergedDir_kernel.cuh"
#include "weightedCoeffDiagHVHT_sftr_kernel.cuh"
#include "projection_sftr_opti.cuh"
#include "projection_sf_rec_rec_kernel.cuh"*/
/* RegularSamplingProjector definition */
/*template <typename T>
RegularSamplingProjector_GPU<T>::RegularSamplingProjector_GPU() : Projector<Volume_GPU,Sinogram3D_GPU,T>() {}
 */
/* RegularSamplingProjector definition */
template <typename T>
RegularSamplingProjector_compute_CUDA_mem_GPU<T>::RegularSamplingProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture*  cudaprojectionArchitecture, Volume_GPU<T>* volume) : Projector<Volume_GPU,Sinogram3D_GPU,T>(acquisition, detector,  volume)
{
	this->setCUDAProjectionArchitecture(cudaprojectionArchitecture);
	cout << "********** Start Constant Copy **********" << endl;
	cout << "Projection Constant Copy on device n° " << 0 << endl;
	checkCudaErrors(cudaSetDevice(0));
	this->copyConstantGPU();
	cout << "********** End Projection Constant Copy **********" << endl;
}

template <typename T>
RegularSamplingProjector_compute_CUDA_mem_GPU<T>::~RegularSamplingProjector_compute_CUDA_mem_GPU(){}


template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_GPU<T>::doProjection(Sinogram3D_GPU<T>* estimatedSinogram,Volume_GPU<T> *volume)
{
	std::cout << "\tRegular Sampling Projection all on GPU" << std::endl;
	this->setVolume(volume);


	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	checkCudaErrors(cudaSetDevice(0));

	dim3 dimBlock(this->getCUDAProjectionArchitecture()->getXThreadNb(),this->getCUDAProjectionArchitecture()->getYThreadNb(),this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	dim3 dimGrid(estimatedSinogram->getUSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getXThreadNb(),estimatedSinogram->getVSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getYThreadNb(),estimatedSinogram->getProjectionSinogramNb()/this->getCUDAProjectionArchitecture()->getProjectionThreadNb());


	//VOLUME MEMORY ALLOCATION
	//Create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(T), 0, 0, 0, cudaChannelFormatKindFloat);
	struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	volume_cu_array_size= make_cudaExtent(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb);
	checkCudaErrors(cudaMalloc3DArray(&volume_cu_array, &channelDesc,volume_cu_array_size ) );


	//copy data to 3D array
	cudaMemcpy3DParms VolumeParams = {0};
	VolumeParams.srcPos = make_cudaPos(0,0,0);
	VolumeParams.dstPos = make_cudaPos(0,0,0);
	VolumeParams.dstArray = volume_cu_array;
	VolumeParams.extent = volume_cu_array_size;
	VolumeParams.kind = cudaMemcpyDeviceToDevice;

	T* volumeData = this->getVolume()->getVolumeData();
	T* estimatedSinogramData = estimatedSinogram->getDataSinogram();

	VolumeParams.srcPtr = make_cudaPitchedPtr(volumeData, VolumeParams.extent.width*sizeof(T), VolumeParams.extent.width, VolumeParams.extent.height);

	checkCudaErrors(cudaMemcpy3D(&VolumeParams));

	// set texture parameters
	volume_tex.addressMode[0] = cudaAddressModeBorder;
	volume_tex.addressMode[1] = cudaAddressModeBorder;
	volume_tex.addressMode[2] = cudaAddressModeBorder;
	volume_tex.filterMode = cudaFilterModeLinear;
	volume_tex.normalized = false; // access with normalized texture coordinates

	checkCudaErrors(cudaBindTextureToArray(volume_tex, volume_cu_array, channelDesc));

	projection_ERB_kernel_v0_UM<<< dimGrid, dimBlock>>>(estimatedSinogramData);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaUnbindTexture(volume_tex));

	checkCudaErrors(cudaFreeArray(volume_cu_array));

}


template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_GPU<T>::EnableP2P(){}

template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_GPU<T>::DisableP2P(){}

template<typename T>
CUDAProjectionArchitecture* RegularSamplingProjector_compute_CUDA_mem_GPU<T>::getCUDAProjectionArchitecture() const
{
	return this->cudaprojectionArchitecture;
}

template <typename T>
__host__ void RegularSamplingProjector_compute_CUDA_mem_GPU<T>::copyConstantGPU()
{
	unsigned long int projectionNb = (this->getAcquisition())->getProjectionNb();
	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();
	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();

	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	unsigned long int uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	unsigned long int vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
	float GammaIOcylinderC = this->getGammaIOcylinderC();

	cudaMemcpyToSymbol(xVolumeCenterPixel_GPU,&xVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(yVolumeCenterPixel_GPU,&yVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(zVolumeCenterPixel_GPU,&zVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(xVolumePixelSize_GPU,&xVolumePixelSize,sizeof(float));
	cudaMemcpyToSymbol(xVolumePixelNb_GPU,&xVolumePixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(yVolumePixelNb_GPU,&yVolumePixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(zVolumePixelNb_GPU,&zVolumePixelNb,sizeof(unsigned long int));

	cudaMemcpyToSymbol(focusDetectorDistance_GPU,&fdd,sizeof(float));
	cudaMemcpyToSymbol(focusObjectDistance_GPU,&fod,sizeof(float));

	cudaMemcpyToSymbol(uDetectorCenterPixel_GPU,&uDetectorCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(vDetectorCenterPixel_GPU,&vDetectorCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(uDetectorPixelSize_GPU,&uDetectorPixelSize,sizeof(float));
	cudaMemcpyToSymbol(vDetectorPixelSize_GPU,&vDetectorPixelSize,sizeof(float));
	cudaMemcpyToSymbol(uDetectorPixelNb_GPU,&uDetectorPixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(vDetectorPixelNb_GPU,&vDetectorPixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(projectionNb_GPU,&projectionNb,sizeof(unsigned long int));

	cudaMemcpyToSymbol(alphaIOcylinderC_GPU,this->getAlphaIOcylinderC(),projectionNb*sizeof(float));
	cudaMemcpyToSymbol(betaIOcylinderC_GPU,this->getBetaIOcylinderC(),projectionNb*sizeof(float));
	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&GammaIOcylinderC,sizeof(float));

}

template<typename T>
void RegularSamplingProjector_compute_CUDA_mem_GPU<T>::setCUDAProjectionArchitecture(CUDAProjectionArchitecture*  cudaprojectionArchitecture)
{
	this->cudaprojectionArchitecture =  cudaprojectionArchitecture;
}

#include "Projector_instances_GPU.cu"