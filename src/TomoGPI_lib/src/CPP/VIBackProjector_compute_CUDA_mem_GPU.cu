/*
 * VIBackProjector_compute_CUDA_mem_GPU.cu
 *
  *      Author: gac
 */

#include "BackProjector_GPU.cuh"
#include "BackProjector_kernel.cuh"

/* VIBackProjector_GPU definition */
template<typename T>
VIBackProjector_compute_CUDA_mem_GPU<T>::VIBackProjector_compute_CUDA_mem_GPU(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture,Volume_GPU<T>* volume,char fdk) : BackProjector<Volume_GPU,Sinogram3D_GPU,T>(acquisition, detector, volume, fdk)
{
	this->setCUDABProjectionArchitecture(cudabackprojectionArchitecture);
	cout << "********** Start Constant Copy **********" << endl;
	cout << "BackProjection Constant Copy on device n° " << 0 << endl;
	checkCudaErrors(cudaSetDevice(0));
	//	cudaDeviceReset();
	this->copyConstantGPU();
	cout << "********** End BackProjection Constant Copy **********" << endl;
}

template<typename T>
VIBackProjector_compute_CUDA_mem_GPU<T>::~VIBackProjector_compute_CUDA_mem_GPU(){}

template<typename T>
void VIBackProjector_compute_CUDA_mem_GPU<T>::doBackProjection(Volume_GPU<T> *estimatedVolume,Sinogram3D_GPU<T>* sinogram)
{
	std::cout << "\tVI BackProjection all on GPU" << std::endl;

	unsigned long int uSinogramPixelNb = sinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = sinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramPixelNb = sinogram->getProjectionSinogramNb();

	this->setVolume(estimatedVolume);

	T* sinogramData = sinogram->getDataSinogram();

	cudaChannelFormatDesc channelDesc;
	cudaArray *sino_cu_3darray;
	cudaMemcpy3DParms myparms_sino_3Darray = {0};

	//Set device
	//CUDA_VISIBLE_DEVICES=0;
	checkCudaErrors(cudaSetDevice(0));


	//Decoupage en thread
	dim3 dimBlock(this->getCUDABProjectionArchitecture()->getXThreadNb(),this->getCUDABProjectionArchitecture()->getYThreadNb(),this->getCUDABProjectionArchitecture()->getZThreadNb());
	dim3 dimGrid(this->getVolume()->getXVolumePixelNb()/this->getCUDABProjectionArchitecture()->getXThreadNb(), this->getVolume()->getYVolumePixelNb()/this->getCUDABProjectionArchitecture()->getYThreadNb(), this->getVolume()->getZVolumePixelNb()/16);

	//std::cout << "dimBlock.x" <<dimBlock.x << "dimBlock.y" <<dimBlock.y << "dimBlock.z" <<dimBlock.z << std::endl;
	//std::cout << "dimGrid.x" <<dimGrid.x <<"dimGrid.y" << dimGrid.y << "dimGrid.z" <<dimGrid.z << std::endl;

	//Mise des sinogram en texture 2D layered
	channelDesc = cudaCreateChannelDesc(8*sizeof(T), 0, 0, 0, cudaChannelFormatKindFloat);

	checkCudaErrors(cudaMalloc3DArray(&sino_cu_3darray, &channelDesc, make_cudaExtent(uSinogramPixelNb,vSinogramPixelNb,projectionSinogramPixelNb), cudaArrayLayered));

	myparms_sino_3Darray.srcPos = make_cudaPos(0,0,0);
	myparms_sino_3Darray.dstPos = make_cudaPos(0,0,0);

	myparms_sino_3Darray.srcPtr = make_cudaPitchedPtr(sinogramData, uSinogramPixelNb*sizeof(T), uSinogramPixelNb,vSinogramPixelNb);

	myparms_sino_3Darray.dstArray = sino_cu_3darray;
	myparms_sino_3Darray.extent = make_cudaExtent(uSinogramPixelNb,vSinogramPixelNb,projectionSinogramPixelNb);
	myparms_sino_3Darray.kind = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&myparms_sino_3Darray));

	sinogram_tex0.addressMode[0] = cudaAddressModeBorder;
	sinogram_tex0.addressMode[1] = cudaAddressModeBorder;
	sinogram_tex0.filterMode = cudaFilterModeLinear;
	sinogram_tex0.normalized = false;    // access with normalized texture coordinates

	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(sinogram_tex0, sino_cu_3darray, channelDesc));

	/*T* vol;
  		cout << "Allocation dans Image3D_GPU : "  << sizeof(T)*this->getVolume()->getVolumeImage()->getDataImageSize() << "projectionSinogramPixelNb" << projectionSinogramPixelNb << endl;
  		checkCudaErrors(cudaMalloc((void **)&vol, sizeof(T)*this->getVolume()->getVolumeImage()->getDataImageSize()));
  		checkCudaErrors(cudaDeviceSynchronize());*/

	backprojection_VIB_kernel_v1_16reg_UM<<< dimGrid, dimBlock>>>(this->getVolume()->getVolumeData(),projectionSinogramPixelNb);
	//backprojection_VIB_kernel_v1_16reg_UM<<< dimGrid, dimBlock>>>(vol,projectionSinogramPixelNb);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaUnbindTexture(sinogram_tex0));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFreeArray(sino_cu_3darray));


}

template <typename T>
void VIBackProjector_compute_CUDA_mem_GPU<T>::EnableP2P(){}

template <typename T>
void VIBackProjector_compute_CUDA_mem_GPU<T>::DisableP2P(){}

template<typename T>
CUDABProjectionArchitecture* VIBackProjector_compute_CUDA_mem_GPU<T>::getCUDABProjectionArchitecture() const
{
	return this->cudabackprojectionArchitecture;
}

template <typename T>
__host__ void VIBackProjector_compute_CUDA_mem_GPU<T>::copyConstantGPU()
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
	
	float gammaIOcylinderC= this->getGammaIOcylinderC();
	float alphaC= this->getAlphaC();
	float betaC= this->getBetaC();

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
	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&gammaIOcylinderC,sizeof(float));

	cudaMemcpyToSymbol(alphaC_GPU,&alphaC,sizeof(float));
	cudaMemcpyToSymbol(betaC_GPU,&betaC,sizeof(float));
}

template<typename T>
void VIBackProjector_compute_CUDA_mem_GPU<T>::setCUDABProjectionArchitecture(CUDABProjectionArchitecture*  cudabackprojectionArchitecture)
{
	this->cudabackprojectionArchitecture =  cudabackprojectionArchitecture;
}

#include "BackProjector_instances_GPU.cu"
