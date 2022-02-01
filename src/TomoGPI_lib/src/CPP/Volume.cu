/*
 * Volume.cu
 *
 *      Author: gac
 */

#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "GPUConstant.cuh"

template <typename T>
Volume<T>::Volume() : xVolumeSize(0), yVolumeSize(0), zVolumeSize(0), xVolumePixelNb(0), yVolumePixelNb(0), zVolumePixelNb(0), xVolumePixelSize(0), yVolumePixelSize(0), zVolumePixelSize(0), xVolumeCenterPixel(0), yVolumeCenterPixel(0), zVolumeCenterPixel(0), xVolumeStartPixel(0), yVolumeStartPixel(0), zVolumeStartPixel(0),volumeImage(0),segmentation(0),contours(0),number_classes(0),m_classes(0),v_classes(0),energy_singleton(0),gamma_potts(0){}

template <typename T>
Volume<T>::Volume(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb) : xVolumeSize(xVolumeSize), yVolumeSize(yVolumeSize), zVolumeSize(zVolumeSize), xVolumePixelNb(xVolumePixelNb), yVolumePixelNb(yVolumePixelNb), zVolumePixelNb(zVolumePixelNb), xVolumeStartPixel(0), yVolumeStartPixel(0), zVolumeStartPixel(0), volumeImage(0),segmentation(0),contours(0),number_classes(0),m_classes(0),v_classes(0),energy_singleton(0),gamma_potts(0)
{
	xVolumePixelSize = xVolumeSize/xVolumePixelNb;
	yVolumePixelSize = yVolumeSize/yVolumePixelNb;
	zVolumePixelSize = xVolumePixelSize;
	xVolumeCenterPixel = xVolumePixelNb/2.0 - 0.5;
	yVolumeCenterPixel = yVolumePixelNb/2.0 - 0.5;
	zVolumeCenterPixel = zVolumePixelNb/2.0 - 0.5;
	//	this->volumeImage = new Image3D<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb);
}

template <typename T>
Volume<T>::~Volume()
{
	//	delete volumeImage;
}

template <typename T>
Volume<T>&  Volume<T>::operator=(const Volume<T> &volume)
{
	this->xVolumeSize = volume.xVolumeSize;
	this->yVolumeSize = volume.yVolumeSize;
	this->zVolumeSize = volume.zVolumeSize;

	this->xVolumePixelNb = volume.xVolumePixelNb;
	this->yVolumePixelNb = volume.yVolumePixelNb;
	this->zVolumePixelNb = volume.zVolumePixelNb;

	this->xVolumePixelSize = volume.xVolumePixelSize;
	this->yVolumePixelSize = volume.yVolumePixelSize;
	this->zVolumePixelSize = volume.zVolumePixelSize;

	this->xVolumeCenterPixel = volume.xVolumeCenterPixel;
	this->yVolumeCenterPixel = volume.yVolumeCenterPixel;
	this->zVolumeCenterPixel = volume.zVolumeCenterPixel;

	this->xVolumeStartPixel = volume.xVolumeStartPixel;
	this->yVolumeStartPixel = volume.yVolumeStartPixel;
	this->zVolumeStartPixel = volume.zVolumeStartPixel;
	this->volumeImage = volume.volumeImage;

	return *this;
}

template <typename T>
bool Volume<T>::isSameSize(Volume<T>* volume2) const
{
	return (this->getVolumeImage()->isSameSize(volume2->getVolumeImage()));
}

template <typename T>
float Volume<T>::getXVolumeSize() const
{
	return xVolumeSize;
}

template <typename T>
float Volume<T>::getYVolumeSize() const
{
	return yVolumeSize;
}

template <typename T>
float Volume<T>::getZVolumeSize() const
{
	return zVolumeSize;
}

template <typename T>
void Volume<T>::setXVolumeSize(float xVolumeSize)
{
	this->xVolumeSize = xVolumeSize;
}

template <typename T>
void  Volume<T>::setYVolumeSize(float yVolumeSize)
{
	this->yVolumeSize = yVolumeSize;
}

template <typename T>
void  Volume<T>::setZVolumeSize(float zVolumeSize)
{
	this->zVolumeSize = zVolumeSize;
}

template <typename T>
unsigned long int Volume<T>::getXVolumePixelNb() const
{
	return xVolumePixelNb;
}

template <typename T>
unsigned long int Volume<T>::getYVolumePixelNb() const
{
	return yVolumePixelNb;
}

template <typename T>
unsigned long int Volume<T>::getZVolumePixelNb() const
{
	return zVolumePixelNb;
}

template <typename T>
void Volume<T>::setXVolumePixelNb(unsigned long int xVolumePixelNb)
{
	this->xVolumePixelNb = xVolumePixelNb;
}

template <typename T>
void Volume<T>::setYVolumePixelNb(unsigned long int yVolumePixelNb)
{
	this->yVolumePixelNb = xVolumePixelNb;
}

template <typename T>
void Volume<T>::setZVolumePixelNb(unsigned long int zVolumePixelNb)
{
	this->zVolumePixelNb = zVolumePixelNb;
}

template <typename T>
float Volume<T>::getXVolumePixelSize() const
{
	return xVolumePixelSize;
}

template <typename T>
float Volume<T>::getYVolumePixelSize() const
{
	return yVolumePixelSize;
}

template <typename T>
float Volume<T>::getZVolumePixelSize() const
{
	return zVolumePixelSize;
}

template <typename T>
void Volume<T>::setXVolumePixelSize(float xVolumePixelSize)
{
	this->xVolumePixelSize = xVolumePixelSize;
}

template <typename T>
void Volume<T>::setYVolumePixelSize(float yVolumePixelSize)
{
	this->yVolumePixelSize = yVolumePixelSize;
}

template <typename T>
void Volume<T>::setZVolumePixelSize(float zVolumePixelSize)
{
	this->zVolumePixelSize = zVolumePixelSize;
}

template <typename T>
float Volume<T>::getXVolumeCenterPixel() const
{
	return xVolumeCenterPixel;
}

template <typename T>
float Volume<T>::getYVolumeCenterPixel() const
{
	return yVolumeCenterPixel;
}

template <typename T>
float Volume<T>::getZVolumeCenterPixel() const
{
	return zVolumeCenterPixel;
}

template <typename T>
void Volume<T>::setXVolumeCenterPixel(float xVolumeCenterPixel)
{
	this->xVolumeCenterPixel = xVolumeCenterPixel;
}

template <typename T>
void Volume<T>::setYVolumeCenterPixel(float yVolumeCenterPixel)
{
	this->yVolumeCenterPixel = yVolumeCenterPixel;
}

template <typename T>
void Volume<T>::setZVolumeCenterPixel(float zVolumeCenterPixel)
{
	this->zVolumeCenterPixel = zVolumeCenterPixel;
}

template <typename T>
float Volume<T>::getXVolumeStartPixel() const
{
	return xVolumeStartPixel;
}

template <typename T>
float Volume<T>::getYVolumeStartPixel() const
{
	return yVolumeStartPixel;
}

template <typename T>
float Volume<T>::getZVolumeStartPixel() const
{
	return zVolumeStartPixel;
}

template <typename T>
void Volume<T>::setXVolumeStartPixel(float xVolumeStartPixel)
{
	this->xVolumeStartPixel = xVolumeStartPixel;
}

template <typename T>
void Volume<T>::setYVolumeStartPixel(float yVolumeStartPixel)
{
	this->yVolumeStartPixel = yVolumeStartPixel;
}

template <typename T>
void Volume<T>::setZVolumeStartPixel(float zVolumeStartPixel)
{
	this->zVolumeStartPixel = zVolumeStartPixel;
}

template <typename T>
void Volume<T>::saveVolume(string fileName)
{
	this->getVolumeImage()->saveImage(fileName);
}

template <typename T>
void Volume<T>::saveVolumeIter(string fileName)
{
	this->getVolumeImage()->saveImageIter(fileName);
}

template <typename T>
void Volume<T>::saveMiddleSliceVolume(string fileName)
{
	this->getVolumeImage()->saveMiddleSliceImage(fileName);
}

template <typename T>
void Volume<T>::loadVolume(string fileName)
{
	this->getVolumeImage()->loadImage(fileName);
}

template <typename T>
void Volume<T>::loadVolume(string fileName, unsigned long int offSet)
{
	this->getVolumeImage()->loadImage(fileName, offSet);
}

template <typename T>
Image3D<T>* Volume<T>::getVolumeImage() const
{
	return volumeImage;
}

template <typename T>
void Volume<T>::setVolumeImage(Image3D<T>* volumeImage)
{
	this->volumeImage = volumeImage;
}


template <typename T>
void Volume<T>::setVolumeData(T* dataImage)
{
	this->getVolumeImage()->setImageData(dataImage);
}

template <typename T>
T* Volume<T>::getVolumeData() const
{
	return this->getVolumeImage()->getImageData();
}


/*
void Volume<T>::InitVolume(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram){}

void Volume<T>::InitVolume_InitIter(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram){}

void Volume<T>::InitVolume_InitSG(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram){}

void Volume<T>::InitUpdate(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume1, Volume<T>* volume2, Sinogram3D<T>* sinogram){}
*/

#include "Volume_instances.cu"