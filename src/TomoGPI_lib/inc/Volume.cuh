/*
 * Volume.cuh
 *
 *      Author: gac
 */

#ifndef VOLUME_HPP_
#define VOLUME_HPP_

#include <iostream>
#include <omp.h>
#include "FieldOfView.hpp"
#include "Image3D.cuh"
#include "ieeehalfprecision.hpp"
#include "ComputingArchitecture.cuh"

template<typename T> class Volume{

public:

	Volume();
	Volume(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb);
	~Volume();

	Volume(const Volume<T>& volumeToCopy);
	Volume & operator=(const Volume<T> &volume);

	bool isSameSize(Volume<T>* volume2) const;// Test if current volume and volume2 are the same size

	/* Physical Volume Parameters */
	float getXVolumeSize() const; // Get size of reconstruction volume in X
	float getYVolumeSize() const; // Get size of reconstruction volume in Y
	float getZVolumeSize() const; // Get size of reconstruction volume in Z
	void setXVolumeSize(float xVolumeSize); // Set size of reconstruction volume in X
	void setYVolumeSize(float yVolumeSize); // Set size of reconstruction volume in Y
	void setZVolumeSize(float zVolumeSize); // Set size of reconstruction volume in Z

	/* Discrete Volume Parameters */
	unsigned long int getXVolumePixelNb() const; // Get number of reconstruction volume pixel in X
	unsigned long int getYVolumePixelNb() const; // Get number of reconstruction volume pixel in Y
	unsigned long int getZVolumePixelNb() const; // Get number of reconstruction volume pixel in Z
	void setXVolumePixelNb(unsigned long int xVolumePixelNb); // Set number of reconstruction volume pixel in X
	void setYVolumePixelNb(unsigned long int yVolumePixelNb); // Set number of reconstruction volume pixel in Y
	void setZVolumePixelNb(unsigned long int zVolumePixelNb); // Set number of reconstruction volume pixel in Z

	float getXVolumePixelSize() const; // Get reconstruction volume pixel size in X
	float getYVolumePixelSize() const; // Get reconstruction volume voxel size in Y
	float getZVolumePixelSize() const; // Get reconstruction volume voxel size in Z
	void setXVolumePixelSize(float xVolumePixelSize); // Set reconstruction volume voxel size in X
	void setYVolumePixelSize(float yVolumePixelSize); // Set reconstruction volume voxel size in Y
	void setZVolumePixelSize(float zVolumePixelSize); // Set reconstruction volume voxel size in Z

	float getXVolumeCenterPixel() const; // Get position of volume center pixel in X
	float getYVolumeCenterPixel() const; // Get position of volume center pixel in Y
	float getZVolumeCenterPixel() const; // Get position of volume center pixel in Z
	void setXVolumeCenterPixel(float xVolumeCenterPixel); // Set position of volume center pixel in X
	void setYVolumeCenterPixel(float yVolumeCenterPixel); // Set position of volume center pixel in Y
	void setZVolumeCenterPixel(float zVolumeCenterPixel); // Set position of volume center pixel in Z

	float getXVolumeStartPixel() const; // Get volume start pixel in X
	float getYVolumeStartPixel() const; // Get volume start pixel in Y
	float getZVolumeStartPixel() const; // Get volume start pixel in Z
	void setXVolumeStartPixel(float xVolumeStartPixel); // Set volume start pixel in X
	void setYVolumeStartPixel(float yVolumeStartPixel); // Set volume start pixel in Y
	void setZVolumeStartPixel(float zVolumeStartPixel); // Set volume start pixel in Z

	Image3D<T>* getVolumeImage() const; // Get volume image
	T* getVolumeData() const; // Get volume data
	void setVolumeData(T* dateImage); // Set volume data
	void setVolumeImage(Image3D<T>* volumeImage); // Set volume image

	void saveVolume(string fileName); // Save volume
	void saveVolumeIter(string fileName); // Save volume
	void saveMiddleSliceVolume(string fileName); // Save middle slice volume
	void loadVolume(string fileName); // Load volume
	void loadVolume(string fileName,unsigned long int offSet); // Load volume from offSet

	
	//Debug
	//void saveVolumeDebug(string fileName); // Save volume
	
	/*void InitVolume(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitVolume_InitIter(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitVolume_InitSG(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitUpdate(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume1, Volume<T>* volume2, Sinogram3D<T>* sinogram);*/
private:
	/* Physical Volume Parameters */
	float xVolumeSize; // X size of reconstruction volume
	float yVolumeSize; // Y size of reconstruction volume
	float zVolumeSize; // Z size of reconstruction volume

	/* Discrete Volume Parameters */
	unsigned long int xVolumePixelNb; // X number of volume pixel
	unsigned long int yVolumePixelNb; // Y number of volume pixel
	unsigned long int zVolumePixelNb; // Z number of volume pixel

	float xVolumePixelSize; // X volume pixel size
	float yVolumePixelSize; // Y volume pixel size
	float zVolumePixelSize; // Z volume pixel size

	float xVolumeCenterPixel; // Position of volume center pixel in X
	float yVolumeCenterPixel; // Position of volume center pixel in Y
	float zVolumeCenterPixel; // Position of volume center pixel in Z

	float xVolumeStartPixel; // X volume start pixel
	float yVolumeStartPixel; // Y volume start pixel
	float zVolumeStartPixel; // Z volume start pixel

	Image3D<T>* volumeImage;

	// Gauss-Markov-Potts
	Image3D<int>* segmentation;//labels
	Image3D<bool>* contours;//indicate contours
	int number_classes;// number of classes
	double* m_classes;// means in the classes
	double* v_classes;//variances of the classes
	double* energy_singleton;//energy of singletons
	double gamma_potts;//Potts coefficient
};



#endif /* VOLUME_HPP_ */
