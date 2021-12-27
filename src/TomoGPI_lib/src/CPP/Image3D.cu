/*
 * Image3D.cu
 *
 *      Author: gac
 */


#include "Image3D.cuh"
#include "Image3D_CPU.cuh"
#include "Image3D_GPU.cuh"
//#include "Image3D_MGPU.cuh"
#include "GPUConstant.cuh"
#include "Image3D_GPU_kernel_half.cuh"
#include "Image3D_GPU_kernel.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
//#include "Volume.cuh"

template <typename T>
Image3D<T>::Image3D() : xImagePixelNb(0), yImagePixelNb(0), zImagePixelNb(0),dataImage(0)
{
	this->dataImageSize = xImagePixelNb*yImagePixelNb*zImagePixelNb;
}

template <typename T>
Image3D<T>::Image3D(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, T* dataImage) : xImagePixelNb(xImagePixelNb), yImagePixelNb(yImagePixelNb), zImagePixelNb(zImagePixelNb), dataImage(dataImage)
{
	this->dataImageSize = xImagePixelNb*yImagePixelNb*zImagePixelNb;
}

template <typename T>
Image3D<T>::Image3D(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb) : xImagePixelNb(xImagePixelNb), yImagePixelNb(yImagePixelNb), zImagePixelNb(zImagePixelNb), dataImage(0)
{
	this->dataImageSize = xImagePixelNb*yImagePixelNb*zImagePixelNb;
}

template <typename T>
Image3D<T>::~Image3D(){}

template <typename T>
Image3D<T> & Image3D<T>::operator=(const Image3D<T> &image)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	T* currentImageData = this->getImageData();
	T* imageData = image.getImageData();

	if((xNb == image.getXImagePixelNb()) && (yNb == image.getYImagePixelNb()) && (zNb == image.getZImagePixelNb()))
	{
		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
					currentImageData[x+y*xNb+z*xNb*yNb]=imageData[x+y*xNb+z*xNb*yNb];
	}
	else
	{
		cout << "Image must have the same size" << endl;
		exit(EXIT_FAILURE);
	}

	return *this;
}

template <typename T>
bool Image3D<T>::isSameSize(Image3D<T>* image2) const
{
	return (this->getXImagePixelNb() == image2->getXImagePixelNb()) && (this->getYImagePixelNb() == image2->getYImagePixelNb()) && (this->getZImagePixelNb() == image2->getZImagePixelNb());
}

template <typename T>
unsigned long int Image3D<T>::getXImagePixelNb() const
{
	return xImagePixelNb;
}

template <typename T>
unsigned long int Image3D<T>::getYImagePixelNb() const
{
	return yImagePixelNb;
}

template <typename T>
unsigned long int Image3D<T>::getZImagePixelNb() const
{
	return zImagePixelNb;
}

template <typename T>
T* Image3D<T>::getImageData() const
{
	return dataImage;
}

template <typename T>
void Image3D<T>::setImageData(T* dataImage_ptr)
{
	this->dataImage=dataImage_ptr;
}

template <typename T>
unsigned long int Image3D<T>::getDataImageSize() const
{
	return dataImageSize;
}

template <typename T>
void Image3D<T>::setXImagePixelNb(unsigned long int xImagePixelNb)
{
	this->xImagePixelNb = xImagePixelNb;
	setDataImageSize(this->xImagePixelNb,this->yImagePixelNb,this->zImagePixelNb);
}

template <typename T>
void Image3D<T>::setYImagePixelNb(unsigned long int yImagePixelNb)
{
	this->yImagePixelNb = yImagePixelNb;
	setDataImageSize(this->xImagePixelNb,this->yImagePixelNb,this->zImagePixelNb);
}

template <typename T>
void Image3D<T>::setZImagePixelNb(unsigned long int zImagePixelNb)
{
	this->zImagePixelNb = zImagePixelNb;
	setDataImageSize(this->xImagePixelNb,this->yImagePixelNb,this->zImagePixelNb);
}

template <typename T>
void Image3D<T>::setDataImageSize(unsigned long int xImagePixelNb,unsigned long int yImagePixelNb,unsigned long int zImagePixelNb)
{
	this->xImagePixelNb = xImagePixelNb;
	this->yImagePixelNb = yImagePixelNb;
	this->zImagePixelNb = zImagePixelNb;
	this->dataImageSize = this->xImagePixelNb*this->yImagePixelNb*this->zImagePixelNb;
}

template <typename T>
void Image3D<T>::saveImage(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out | ios::binary);


	if (imageFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " image" << endl;
		cout << "size : "<< sizeof(T)*this->getDataImageSize() <<endl;
		cout << "ptr : "<<this->getImageData()<<endl;
		imageFile.write((char*)this->getImageData(),sizeof(T)*this->getDataImageSize());
		imageFile.close();
		string name = "chmod 774 ";
		i=system((name + fileName.c_str()).c_str());
		cout << i << "Image saved in " << fileName << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D<T>::saveImageIter(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out | ios::binary);


	if (imageFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " image" << endl;
		cout << "size : "<< sizeof(T)*this->getDataImageSize() <<endl;
		cout << "ptr : "<<this->getImageData()<<endl;
		imageFile.write((char*)this->getImageData(),sizeof(T)*this->getDataImageSize());
		imageFile.close();
		string name = "chmod 774 ";
		i=system((name + fileName.c_str()).c_str());
		cout << i << "Image saved in " << fileName << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D<T>::saveMiddleSliceImage(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out |ios::app | ios::binary);

	if (imageFile.is_open())
	{
		int i;
		cout << "\tSaving " << fileName << " image" << endl;
		imageFile.write((char*)(&(this->getImageData()[this->getZImagePixelNb()/2*this->getXImagePixelNb()*this->getYImagePixelNb()])),sizeof(T)*this->getXImagePixelNb()*this->getYImagePixelNb());
		imageFile.close();
		string name = "chmod 774 ";
		i=system((name + fileName.c_str()).c_str());
		cout <<   "\tImage saved in " << fileName << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D<T>::loadImage(string fileName)
{
	ifstream imageFile;
	imageFile.open(fileName.c_str(), ios::in|ios::binary);

	if (imageFile.is_open())
	{
		cout << "Loading " << fileName << " Image "<< endl;
		cout << "size : "<< sizeof(T)*this->getDataImageSize() << endl;
		cout << "ptr :" << this->getImageData() << endl;
		imageFile.read ((char*)this->getImageData(), sizeof(T)*this->getDataImageSize());
		imageFile.close();
		cout << "Image " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file" << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D<T>::loadImage(string fileName, unsigned long int offSet)
{
	ifstream imageFile;
	char *tmp;
	imageFile.open(fileName.c_str(), ios::in|ios::binary);
	tmp = (char *)malloc(offSet*sizeof(char));
	if (imageFile.is_open())
	{
		cout << "Loading " << fileName << " image od size" << sizeof(T)*this->getDataImageSize() << endl;
		imageFile.read (tmp, sizeof(T)*offSet);
		imageFile.read ((char*)this->getImageData(), sizeof(T)*this->getDataImageSize());
		imageFile.close();

		cout << "Image " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

#include "Image3D_instances.cu"