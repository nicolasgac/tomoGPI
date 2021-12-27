/*
 * Iter3D_compute.cu
 *
 *      Author: gac
 */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
//#include "Iter3D.MGPUcuh"

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_CUDA(string workdirectory) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workdirectory)
{

	this->cudaArchitectureSino = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->cudaArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDAProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudabackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDABProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION

	float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

	this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,this->cudaArchitectureVolume)); 
	this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->cudaprojectionArchitecture,this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->cudabackprojectionArchitecture,this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_CUDA(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workdirectory,configComputeArchitecture_file)
{
	this->cudaArchitectureSino = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->cudaArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDAProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudabackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDABProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaArchitectureSino->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_sino());
	this->cudaArchitectureVolume->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_vol());
	this->cudaprojectionArchitecture->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_proj());
	this->cudabackprojectionArchitecture->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_back());
this->cudaprojectionArchitecture->setProjectionStreamsNb(this->getConfigComputeArchitectureFile()->getprojectionStreamsNb());
this->cudabackprojectionArchitecture->setBProjectionStreamsNb(this->getConfigComputeArchitectureFile()->getbackprojectionStreamsNb());

float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,this->getCUDAArchitectureVolume())); // VOLUME OBJECT CREATION
this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->getCUDAProjectionArchitecture(),this->getVolume())); // PROJECTION OBJECT CREATION
this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->getCUDABProjectionArchitecture(),this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::~Iter3D_compute_CUDA(){}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::Init1_SG(V<T>* volume,S<T>* sino)
{
/*
this->getProjector()->EnableP2P();
volume->InitVolume_InitSG(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, sino);
sino->InitSinogram3D(this->getAcquisition(), this->getDetector(), this->getCUDABProjectionArchitecture(),volume, sino, 0);
*/
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::Init2_SG(V<T>* volume,V<T>* dJ,S<T>* sino)
{
/*
dJ->InitVolume_InitSG(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),dJ, sino);
sino->MGPUCopy_Simple(dJ, sino);
// Here in order to do the Back Proj & the Update
sino->InitSinogram3D_InitSG(this->getAcquisition(), this->getDetector(), this->getCUDABProjectionArchitecture(),dJ,sino,0);
dJ->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),dJ, sino);

this->getBackprojector()->CopyConstant();
volume->InitUpdate(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, dJ, sino);
*/
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::Init3_SG(S<T>* sino)
{
/*
sino->InitSinogram3D_v4(this->getAcquisition(), this->getDetector(), this->getCUDABProjectionArchitecture(),dJ,sino, 0);
*/
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CUDAArchitecture* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::getCUDAArchitectureSino() const
{
	return (this->cudaArchitectureSino);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CUDAArchitecture* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::getCUDAArchitectureVolume() const
{
	return (this->cudaArchitectureVolume);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CUDAArchitecture* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::getCUDAArchitecture() const
{
	return (this->cudaArchitecture);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CUDAProjectionArchitecture* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::getCUDAProjectionArchitecture() const
{
	return (this->cudaprojectionArchitecture);
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CUDABProjectionArchitecture* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::getCUDABProjectionArchitecture() const
{
	return (this->cudabackprojectionArchitecture);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
V<T>* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::create_volume()
{
	V<T>* vol;
	vol=new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	return vol;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
S<T>* Iter3D_compute_CUDA<P,BP,R_Huber,R_GG,C,V,S,T>::create_sinogram3D()
{
	S<T>* sino;
	sino=new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	return sino;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_C<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_C(string workDirectory) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workDirectory)
{	
	float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

	this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb)); // 
	this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_C<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_C(string workDirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workDirectory,configComputeArchitecture_file)
{
	float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

	this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb)); // 
	this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_C<P,BP,R_Huber,R_GG,C,V,S,T>::~Iter3D_compute_C(){

}



template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
V<T>* Iter3D_compute_C<P,BP,R_Huber,R_GG,C,V,S,T>::create_volume()
{
	V<T>* vol;
	vol=new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb());
	return vol;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
S<T>* Iter3D_compute_C<P,BP,R_Huber,R_GG,C,V,S,T>::create_sinogram3D()
{
	S<T>* sino;
	sino=new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(), this->cudaArchitectureSino);
	return sino;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_OCL(string workdirectory) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workdirectory)
{	
	this->oclArchitectureSino = &(this->getConfigComputeArchitectureFile()->createOCLArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->oclArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createOCLArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->oclprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createOCLProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->oclbackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createOCLBProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION

	this->oclArchitecture_kind = (this->getConfigComputeArchitectureFile()->getArchitecture());


	float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

	this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb)); 
	this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->oclprojectionArchitecture,this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->oclbackprojectionArchitecture,this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D_compute_OCL(string workDirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>(workDirectory,configComputeArchitecture_file)
{
	this->oclArchitectureSino = &(this->getConfigComputeArchitectureFile()->createOCLArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->oclArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createOCLArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->oclprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createOCLProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->oclbackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createOCLBProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION
	this->oclbackprojectionArchitecture->setArchitecture(configComputeArchitecture_file->getArchitecture());

	float xVolumeSize,yVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumeSize,zVolumePixelNb;

	ConfigCT* configCT_file=this->getConfigCTFile();

	xVolumeSize=configCT_file->getConfigFileField<float>("xVolumeSize");
	yVolumeSize = configCT_file->getConfigFileField<float>("yVolumeSize");
	xVolumePixelNb = configCT_file->getConfigFileField<int>("xVolumePixelNb");
	yVolumePixelNb = configCT_file->getConfigFileField<int>("yVolumePixelNb");
	zVolumeSize = configCT_file->getConfigFileField<float>("zVolumeSize");
	zVolumePixelNb = configCT_file->getConfigFileField<int>("zVolumePixelNb");

	this->setVolume(new V<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb)); // 
	this->setProjector(new P<T>(this->getAcquisition(),this->getDetector(),this->oclprojectionArchitecture,this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP<T>(this->getAcquisition(),this->getDetector(),this->oclbackprojectionArchitecture,this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::~Iter3D_compute_OCL(){}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
V<T>* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::create_volume()
{
	V<T>* vol;
	vol=new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb());
	return vol;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
S<T>* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::create_sinogram3D()
{
	S<T>* sino;
	//sino=new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb());
	sino=new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	return sino;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
OCLArchitecture* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::getOCLArchitectureSino() const
{
	return (this->oclArchitectureSino);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
OCLArchitecture* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::getOCLArchitectureVolume() const
{
	return (this->oclArchitectureVolume);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
OCLArchitecture* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::getOCLArchitecture() const
{
	return (this->oclArchitecture);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
OCLProjectionArchitecture* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::getOCLProjectionArchitecture() const
{
	return (this->oclprojectionArchitecture);
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
OCLBProjectionArchitecture* Iter3D_compute_OCL<P,BP,R_Huber,R_GG,C,V,S,T>::getOCLBProjectionArchitecture() const
{
	return (this->oclbackprojectionArchitecture);
}




#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"
//#include "Iter3D_instances_MGPU.cu"