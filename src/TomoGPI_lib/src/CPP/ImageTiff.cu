#include "Image3D.cuh"
#include "Image3D_CPU.cuh"
#include "GPUConstant.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
#include "Volume.cuh"


/* ImageTiff definition */

template <typename T>
ImageTiff<T>::ImageTiff(unsigned short xImagePixelNb, unsigned short yImagePixelNb) : Image3D_CPU<T>(xImagePixelNb,yImagePixelNb,1){}

template <typename T>
ImageTiff<T>::~ImageTiff(){}

template <typename T>
void ImageTiff<T>::loadTiffImage(string fileName)
{
	ifstream imageFile;
	T tmp[4];
	imageFile.open(fileName.c_str(), ios::in|ios::binary);
	cout << fileName << endl;
	if (imageFile.is_open())
	{
		cout << "Loading " << fileName << " image" << endl;
		imageFile.read ((char*)tmp, sizeof(T)*4);
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

/* ImageCalibration definition */

template <typename T>
ImageCalibration<T>::ImageCalibration(unsigned short xImagePixelNb, unsigned short yImagePixelNb) : ImageTiff<T>(xImagePixelNb,yImagePixelNb){}

template <typename T>
ImageCalibration<T>::~ImageCalibration(){}

template <typename T>
double ImageCalibration<T>::getBlackMeanValue()
{
	int x,y;
	double imageMean = 0.0;
	T* tmp = this->getImageData();

	for (y=0;y<this->getYImagePixelNb()/3;y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageMean+=tmp[x+y*this->getXImagePixelNb()];

	imageMean /= (double)this->getXImagePixelNb()*this->getYImagePixelNb()/3.0;

	return imageMean;
}

template <typename T>
double ImageCalibration<T>::getGrayMeanValue()
{
	int x,y;
	double imageMean = 0.0;
	T* tmp = this->getImageData();

	for (y=this->getYImagePixelNb()/3;y<2*this->getYImagePixelNb()/3;y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageMean+=tmp[x+y*this->getXImagePixelNb()];

	imageMean /= (double)this->getXImagePixelNb()*this->getYImagePixelNb()/3.0;

	return imageMean;
}

template <typename T>
double ImageCalibration<T>::getWhiteMeanValue(unsigned int CalibrationEnergyNb)
{
	int x,y;
	double imageMean = 0.0;
	T* tmp = this->getImageData();

	for (y=(CalibrationEnergyNb-1)*this->getYImagePixelNb()/CalibrationEnergyNb;y<this->getYImagePixelNb();y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageMean+=tmp[x+y*this->getXImagePixelNb()];

	imageMean /= (double)this->getXImagePixelNb()*this->getYImagePixelNb()/double(CalibrationEnergyNb);

	return imageMean;
}

template <typename T>
ImageTiff<T>& ImageCalibration<T>::getBlackTiffImage()
{
	int x,y;
	ImageTiff<T>* imageTiff = new ImageTiff<T>(this->getXImagePixelNb(),this->getYImagePixelNb()/3);
	T* tmp = this->getImageData();
	T* imageTiffTmp = imageTiff->getImageData();

	for (y=0;y<this->getYImagePixelNb()/3;y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageTiffTmp[x+y*this->getXImagePixelNb()]=tmp[x+y*this->getXImagePixelNb()];

	return *imageTiff;
}

template <typename T>
ImageTiff<T>& ImageCalibration<T>::getGrayTiffImage()
{
	int x,y;
	ImageTiff<T>* imageTiff = new ImageTiff<T>(this->getXImagePixelNb(),this->getYImagePixelNb()/3);
	T* tmp = this->getImageData();
	T* imageTiffTmp = imageTiff->getImageData();

	for (y=this->getYImagePixelNb()/3;y<2*this->getYImagePixelNb()/3;y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageTiffTmp[x+(y-this->getYImagePixelNb()/3)*this->getXImagePixelNb()]=tmp[x+y*this->getXImagePixelNb()];

	return *imageTiff;
}

template <typename T>
ImageTiff<T>& ImageCalibration<T>::getWhiteTiffImage()
{
	int x,y;
	ImageTiff<T>* imageTiff = new ImageTiff<T>(this->getXImagePixelNb(),this->getYImagePixelNb()/3);
	T* tmp = this->getImageData();
	T* imageTiffTmp = imageTiff->getImageData();

	for (y=2*this->getYImagePixelNb()/3;y<this->getYImagePixelNb();y++)
		for (x=0;x<this->getXImagePixelNb();x++)
			imageTiffTmp[x+(y-2*this->getYImagePixelNb()/3)*this->getXImagePixelNb()]=tmp[x+y*this->getXImagePixelNb()];

	return *imageTiff;
}

#include "Image3D_instances.cu"
#include "Image3D_instances_CPU.cu"