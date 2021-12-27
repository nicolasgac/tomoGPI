/*
 * Sinogram3D.cuh
 *
 *      Author: gac
 */

#ifndef SINOGRAM3D_HPP_
#define SINOGRAM3D_HPP_

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#include "ieeehalfprecision.hpp"
#include "ComputingArchitecture.cuh"
#include "Volume.cuh"
#include "Detector.hpp"

using namespace std;

template<typename T> class Sinogram3D{
public:
	Sinogram3D();
	Sinogram3D(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb);
	Sinogram3D(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb,T* dataSino);
	~Sinogram3D();

	Sinogram3D & operator=(const Sinogram3D &sinogram);

	bool isSameSize(Sinogram3D<T>* sinogram2) const; // Test if current sinogram and sinogram2 are the same size

	unsigned long int getUSinogramPixelNb() const; // Get horizontal sinogram number of pixel
	unsigned long int getVSinogramPixelNb() const; // Get vertical sinogram number of pixel
	unsigned long int getProjectionSinogramNb() const; // Get sinogram number of projections
	unsigned long int getDataSinogramSize() const; // Get data sinogram size

	void setUSinogramPixelNb(unsigned long int uSinogramPixelNb); // Set horizontal sinogram number of pixel
	void setVSinogramPixelNb(unsigned long int vSinogramPixelNb); // Set vertical sinogram number of pixel
	void setProjectionSinogramNb(unsigned long int projectionSinogramNb); // Set sinogram number of projections
	void setDataSinogramSize(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb); // Set data sinogram size

	T* getDataSinogram() const; // Get data sinogram pointer
	void setDataSinogram(T* dataSinogram); // Set data sinogram pointer

	void saveSinogram(string fileName); // Save sinogram
	void loadSinogram(string fileName); // Load sinogram
	void loadSinogram_InitIter(string fileName);

	//debug
	//void saveSinogramDebug(string fileName); // Save sinogram
	//void loadSinogramDebug(string fileName); // Load sinogram
	//void loadSinogramDebug_v2(string fileName); // Load sinogram
	//void loadSinogramDebug_v3(string fileName); // Load sinogram

private:
	unsigned long int uSinogramPixelNb; // U sinogram number of pixel
	unsigned long int vSinogramPixelNb; // U sinogram number of pixel
	unsigned long int projectionSinogramNb; // Sinogram number of projections
	unsigned long int dataSinogramSize; // Data sinogram size
	T* dataSinogram; // Sinogram data

	/*void InitSinogram3D(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk);
	void InitSinoBack(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk);
	void InitSinogram3D_InitSG(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk);
	void InitSinogram3D_v4(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk);
	void MGPUCopy();
	void MGPUCopy_Simple(Volume<T>* volume, Sinogram3D<T>* sinogram);*/
	/*Volume<T>* volume;
	Acquisition* acquisition;
	Detector* detector;
	char fdk;
	Sinogram3D<T>* sinogram;*/
};

#endif /* SINOGRAM3D_HPP_ */
