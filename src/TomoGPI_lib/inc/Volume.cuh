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

	// Gauss-Markov-Potts
	Image3D<int>* getSegmentation();//get segmentation of the volume
	int* getLabels();// get labels of the voxels
	Image3D<bool>* getContours();// get contours of the volume
	bool* getBooleanContours();// get booleans that indicate contours
	double* getMeansClasses();//get means in the classes
	double* getVariancesClasses();//get variances of the classes
	double* getEnergySingleton();//get energies of singleton for each class
	double getGammaPotts();// get Potts coefficient
	int getNumberClasses();// get number of classes

	void setSegmentation(Image3D<int>* segmentation);//set segmentation of the volume
	void setLabels(int* labels);// set labels of the voxels
	void setContours(Image3D<bool>* contours);// set contours
	void setBooleanContours(bool* contours);//set booleans that indicate contours
	void setMeansClasses(double* m_classes);//set means in the classes
	void setVariancesClasses(double* v_classes);//set variances of the classes
	void setEnergySingleton(double* energy_singleton);//set energies of singleton for each class
	void setGammaPotts(double gamma_potts);// set Potts coefficient
	void setNumberClasses(int number_classes);// set number of classes

	// Joint MAP (MGI)
	virtual void maxMeansClassesMGI(double m0, double v0)=0;// MAP for m_classes (MGI)
	virtual void maxVariancesClassesMGI(double alpha0, double beta0)=0;//MAP for v_classes (MGI)
	virtual void maxLabelsMGI(unsigned int numit, double tol)=0;//MAP for labels
	virtual void maxLabelsMGIBlancs()=0;//MAP for voxels "blancs"
	virtual void maxLabelsMGINoirs()=0;//MAP for voxels "noirs"
	virtual double computePottsEnergyMGI()=0;//compute Potts energy
	// Joint MAP (MGM)
	virtual void maxMeansClassesMGM(double m0, double v0)=0;// MAP for m_classes (MGM)
	virtual void maxVariancesClassesMGM(double alpha0, double beta0)=0;//MAP for v_classes (MGM)
	virtual void maxMeansClassesMGMKnownContours(double m0, double v0)=0;// MAP for m_classes (MGM) with known contours
	virtual void maxVariancesClassesMGMKnownContours(double alpha0, double beta0)=0;//MAP for v_classes (MGM) with known contours
	virtual void maxLabelsMGM(unsigned int numit, double tol)=0;//MAP for labels (MGM)
	virtual void maxLabelsMGMBlancs()=0;//MAP for voxels "blancs" (MGM)
	virtual void maxLabelsMGMNoirs()=0;//MAP for voxels "noirs" (MGM)
	virtual double computePottsEnergyMGM()=0;//compute Potts energy (MGM)
	virtual void maxLabelsMGMFixedContours(unsigned int numit, double tol)=0;//MAP for labels (MGM) with fixed contours
	virtual void maxLabelsMGMBlancsFixedContours()=0;//MAP for voxels "blancs" (MGM) with fixed contours
	virtual void maxLabelsMGMNoirsFixedContours()=0;//MAP for voxels "noirs" (MGM) with fixed contours
	virtual double computePottsEnergyMGMFixedContours()=0;//compute Potts energy (MGM) with fixed contours
	virtual void selectContoursVolume()=0;// select contours of the volume
	virtual void selectNoContoursVolume()=0;// select voxels which are not on the contours
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
