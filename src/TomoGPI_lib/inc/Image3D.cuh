/*
 * Image3D.cuh
 *
 *      Author: gac
 */

#ifndef IMAGE3D_HPP_
#define IMAGE3D_HPP_

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#include "ieeehalfprecision.hpp"
#include "ComputingArchitecture.cuh"
#include "Detector.hpp"

using namespace std;

template<typename T> class Image3D{

public:

	Image3D();
	Image3D(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, T* dataImage);
	//Image3D(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,CUDAArchitecture* cudaArchitecture);
	Image3D(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb);
	~Image3D();

	Image3D(const Image3D<T>& imageToCopy); // Copy-constructor

	Image3D & operator=(const Image3D &image);

	bool isSameSize(Image3D<T>* image2) const; // Test if current image and image2 are the same size

	unsigned long int getXImagePixelNb() const; // Get X image number of pixel
	unsigned long int getYImagePixelNb() const; // Get Y image number of pixel
	unsigned long int getZImagePixelNb() const; // Get Z image number of pixel
	unsigned long int getDataImageSize() const; // Get data image size

	void setXImagePixelNb(unsigned long int xImagePixelNb); // Set X image number of pixel
	void setYImagePixelNb(unsigned long int yImagePixelNb); // Set Y image number of pixel
	void setZImagePixelNb(unsigned long int zImagePixelNb); // Set Z image number of pixel
	void setDataImageSize(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb); // Set data image size

	T* getImageData() const; // Get data image pointer
	void setImageData(T* imageData_ptr); // Set data image pointer
	
	void saveImage(string fileName); // Save image
	void saveImageIter(string fileName); // Save image
	void saveMiddleSliceImage(string fileName); // Save middle slice image
	void loadImage(string fileName); // Load image
	void loadImage(string fileName,unsigned long int offSet); // Load image from offSet

	//Debug
	//void saveImageDebug(string fileName); // Save image

private:
	unsigned long int xImagePixelNb; // X image number of pixel
	unsigned long int yImagePixelNb; // Y image number of pixel
	unsigned long int zImagePixelNb; // Z image number of pixel
	unsigned long int dataImageSize; // Data image size

	T* dataImage; // Image data
};

#endif /* IMAGE3D_HPP_ */
