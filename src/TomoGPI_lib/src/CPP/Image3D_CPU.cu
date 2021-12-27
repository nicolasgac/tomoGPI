#include "Image3D_CPU.cuh"
#include "Image3D_CPU_half.cuh"
#include "GPUConstant.cuh"
#include "Image3D_GPU_kernel_half.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Image3D_GPU_kernel.cuh"
#include "Acquisition.hpp"
#include "Volume.cuh"


template <typename T>
Image3D_CPU<T>::Image3D_CPU() : Image3D<T>(){}

template <typename T>
Image3D_CPU<T>::Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, T* dataImage) : Image3D<T>(xImagePixelNb, yImagePixelNb, zImagePixelNb,dataImage){}

template <typename T>
Image3D_CPU<T>::Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture) : Image3D<T>(xImagePixelNb, yImagePixelNb, zImagePixelNb){
	T* dataImageTmp = this->getImageData();
	this->setCUDAArchitecture(cudaArchitecture);
	checkCudaErrors(cudaHostAlloc (&dataImageTmp,this->getDataImageSize()*sizeof(T),cudaHostAllocPortable));
	this->setImageData(dataImageTmp);
	//	this->setImageData(new T[this->getDataImageSize()]());
}

template <typename T>
Image3D_CPU<T>::Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture, T* dataImage) : Image3D<T>(xImagePixelNb, yImagePixelNb, zImagePixelNb,dataImage){}

template <typename T>
Image3D_CPU<T>::Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb) : Image3D<T>(xImagePixelNb, yImagePixelNb, zImagePixelNb)
{
	T* dataImageTmp = this->getImageData();
	checkCudaErrors(cudaHostAlloc (&dataImageTmp,this->getDataImageSize()*sizeof(T),cudaHostAllocPortable));
	this->setImageData(dataImageTmp);
	//	this->setImageData(new T[this->getDataImageSize()]());
}
template <typename T>
Image3D_CPU<T>::~Image3D_CPU()
{
	cudaFreeHost(this->getImageData());
	//	delete this->getImageData();
}

template <>
Image3D_CPU<half>::~Image3D_CPU()
{
	cudaFreeHost(this->getImageData());
	//	delete this->getImageData();
}


template <typename T>
CUDAArchitecture* Image3D_CPU<T>::getCUDAArchitecture() const
{
	return this->cudaArchitecture;
}

template <typename T>
void Image3D_CPU<T>::setCUDAArchitecture(CUDAArchitecture* cudaArchitecture)
{
	this->cudaArchitecture=cudaArchitecture;
}

template <typename T>
Image3D_CPU<T>::Image3D_CPU(const Image3D_CPU<T>& imageToCopy)
{
	unsigned long int xNb,yNb,zNb;
	unsigned long long int voxel;

	this->setXImagePixelNb(imageToCopy.getXImagePixelNb());
	this->setYImagePixelNb(imageToCopy.getYImagePixelNb());
	this->setZImagePixelNb(imageToCopy.getZImagePixelNb());

	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	this->setDataImageSize(xNb,yNb,zNb);
	this->setImageData(new T[this->getDataImageSize()]);

	T* currentImageData = this->getImageData();
	T* imageToCopyData = imageToCopy.getImageData();

# pragma omp parallel for
	for (voxel=0;voxel<xNb*yNb*zNb;voxel++){
		currentImageData[voxel]=imageToCopyData[voxel];
	}

}

template <typename T>
void Image3D_CPU<T>::copyImage3D(Image3D_CPU<T>* imageToCopy)
{
	unsigned long int xNb,yNb,zNb;
	unsigned long long int voxel;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	T* currentImageData = this->getImageData();
	T* imageToCopyData = imageToCopy->getImageData();
#pragma omp parallel for
	for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
		currentImageData[voxel]=imageToCopyData[voxel];
	}
}

template <typename T>
void Image3D_CPU<T>::setImage(T value)
{
	unsigned long int xNb,yNb,zNb;
	unsigned long long int voxel;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	T* currentImageData = this->getImageData();

#pragma omp parallel for
	for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
		currentImageData[voxel]=value;
	}
}

template <typename T>
void Image3D_CPU<T>::addImage(Image3D_CPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]=currentImageData[voxel]+image2Data[voxel];
		}

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::addImage(Image3D_CPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]=currentImageData[voxel]+lambda*image2Data[voxel];
		}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::positiveAddImage(Image3D_CPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]=currentImageData[voxel]+lambda*image2Data[voxel];
			if(currentImageData[voxel]<(T)0.0){
				currentImageData[voxel]=(T)0.0;
			}
		}

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::diffImage(Image3D_CPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]-=image2Data[voxel];
		}

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::diffImage(T lambda, Image3D_CPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]=lambda*currentImageData[voxel]-image2Data[voxel];
		}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::diffImage(Image3D_CPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]=currentImageData[voxel]-lambda*image2Data[voxel];
		}

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::multImage(Image3D_CPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		unsigned long int xNb,yNb,zNb;
		unsigned long long int voxel;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

#pragma omp parallel for
		for(voxel=0;voxel<xNb*yNb*zNb;voxel++){
			currentImageData[voxel]*=image2Data[voxel];
		}

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
double Image3D_CPU<T>::scalarProductImage(Image3D_CPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		double temp=0.0;
		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					temp+=(double)currentImageData[x+y*xNb+z*xNb*yNb]*(double)image2Data[x+y*xNb+z*xNb*yNb];
				}

		return temp;
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
double Image3D_CPU<T>::getImageL1Norm()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double l1Norm = 0.0;

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				l1Norm=l1Norm+(double)abs((double)currentImageData[x+y*xNb+z*xNb*yNb]);

	return l1Norm;
}

template <typename T>
double Image3D_CPU<T>::getImageL2Norm()
{static int i=0;

i++;
unsigned long long int x,y,z,xNb,yNb,zNb;
xNb = this->getXImagePixelNb();
yNb = this->getYImagePixelNb();
zNb = this->getZImagePixelNb();
double l2Norm = 0.0;

T* currentImageData = this->getImageData();
//this->saveImage("Debug/test_"+std::to_string(i)+ ".v");
//cout << xNb <<  yNb << zNb << endl;
for (z=0;z<zNb;z++)
	for (y=0;y<yNb;y++)
		for (x=0;x<xNb;x++){
			double tmp;
			tmp=((double)currentImageData[x+y*xNb+z*xNb*yNb]*(double)currentImageData[x+y*xNb+z*xNb*yNb]);
			l2Norm+=tmp;
			//l2Norm+=(double)(currentImageData[x+y*xNb+z*xNb*yNb]*currentImageData[x+y*xNb+z*xNb*yNb]);

			/*if (std::isnan(l2Norm))
						cout << currentImageData[x+y*xNb+z*xNb*yNb] << " x=" << x << " y=" << y << " z=" << z << endl;*/
		}
if (std::isnan(l2Norm)){
	cout << "Nan"<< endl;
	//this->saveImage("Debug/test_"+std::to_string(i)+ ".v");
}
//cout << "\tl2Norm : "<< l2Norm << endl;
return l2Norm;
}

template <typename T>
double Image3D_CPU<T>::getImageLpNorm(double p)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double lpNorm = 0.0;

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				lpNorm=lpNorm+(double)pow(fabs((double)currentImageData[x+y*xNb+z*xNb*yNb]),(double)p);

	return lpNorm;
}

template <typename T>
double Image3D_CPU<T>::getImageHuberNorm(double threshold)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double tmp = 0;
	double huberNorm =0.0;
	double squareThreshold = threshold*threshold;

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				tmp = double(currentImageData[x+y*xNb+z*xNb*yNb]);
				if(fabs(tmp) < threshold)
					huberNorm += tmp*tmp;
				else if (fabs(tmp) >= threshold)
					huberNorm += 2.0*threshold*fabs(tmp)-squareThreshold;
				else
					huberNorm += tmp;
			}

	return huberNorm;
}

template <typename T>
double Image3D_CPU<T>::getImageMean()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageMean = 0.0;

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				imageMean+=double(currentImageData[x+y*xNb+z*xNb*yNb]);

	imageMean /= (double)this->getDataImageSize();

	return imageMean;
}

template <typename T>
double Image3D_CPU<T>::getImageMeanSquare()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageMeanSquare = 0.0;

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				imageMeanSquare+=double(currentImageData[x+y*xNb+z*xNb*yNb]*currentImageData[x+y*xNb+z*xNb*yNb]);

	imageMeanSquare /= (double)this->getDataImageSize();

	return imageMeanSquare;
}

template <typename T>
double Image3D_CPU<T>::getImageStd()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageStd = 0.0;
	double imageMean = this->getImageMean();

	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				imageStd+= double((imageMean - currentImageData[x+y*xNb+z*xNb*yNb]));

	imageStd /= (double)this->getDataImageSize();

	return imageStd;
}

template <typename T>
void Image3D_CPU<T>::scalarImage(T value)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	T* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				currentImageData[x+y*xNb+z*xNb*yNb]=currentImageData[x+y*xNb+z*xNb*yNb]*value;
}

template <typename T>
void Image3D_CPU<T>::getImageSign(Image3D_CPU<T>* signedImage)
{
	if(this->isSameSize(signedImage))
	{
		unsigned long int xn,yn,zn,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* signedImageData = signedImage->getImageData();

		for (zn=0;zn<zNb;zn++)
			for (yn=0;yn<yNb;yn++)
				for (xn=0;xn<xNb;xn++){
					signedImageData[xn+yn*xNb+zn*xNb*yNb] = fast_sign(currentImageData[xn+yn*xNb+zn*xNb*yNb]);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_CPU<T>::getImageAbsPow(Image3D_CPU<T>* absPowImage, double p)
{
	if(this->isSameSize(absPowImage))
	{
		unsigned long int xn,yn,zn,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		T* currentImageData = this->getImageData();
		T* absPowImageData = absPowImage->getImageData();

		for (zn=0;zn<zNb;zn++)cout << "half float 4" << endl;
		for (yn=0;yn<yNb;yn++)
			for (xn=0;xn<xNb;xn++){
				absPowImageData[xn+yn*xNb+zn*xNb*yNb] = pow(fabs((double)currentImageData[xn+yn*xNb+zn*xNb*yNb]), p);
			}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}


Image3D_CPU_half::Image3D_CPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture) : Image3D_CPU<half>(xImagePixelNb, yImagePixelNb, zImagePixelNb,cudaArchitecture){}


Image3D_CPU_half::Image3D_CPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb) : Image3D_CPU<half>(xImagePixelNb, yImagePixelNb, zImagePixelNb){}


Image3D_CPU_half::~Image3D_CPU_half(){}

Image3D_CPU_half::Image3D_CPU_half(const Image3D_CPU_half& imageToCopy)
{
	unsigned long int x,y,z,xNb,yNb,zNb;

	this->setXImagePixelNb(imageToCopy.getXImagePixelNb());
	this->setYImagePixelNb(imageToCopy.getYImagePixelNb());
	this->setZImagePixelNb(imageToCopy.getZImagePixelNb());

	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	this->setDataImageSize(xNb,yNb,zNb);
	this->setImageData(new half[this->getDataImageSize()]());

	half* currentImageData = this->getImageData();
	half* imageToCopyData = imageToCopy.getImageData();

	float tmp;
	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&imageToCopyData[x+y*xNb+z*xNb*yNb],1);
				singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
			}
}

Image3D_CPU_half & Image3D_CPU_half::operator=(const Image3D_CPU_half &image)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	if((xNb == image.getXImagePixelNb()) && (yNb == image.getYImagePixelNb()) && (zNb == image.getZImagePixelNb()))
	{
		half* currentImageData = this->getImageData();
		half* imageData = image.getImageData();

		float tmp;
		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp,&imageData[x+y*xNb+z*xNb*yNb],1);
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Image must have the same size" << endl;
		exit(EXIT_FAILURE);
	}

	return *this;
}

void Image3D_CPU_half::copyImage3D(Image3D_CPU_half* imageToCopy)
{
	//	unsigned long int x,y,z,xNb,yNb,zNb;


	/*
	T* currentImageData = this->getImageData();
	T* imageToCopyData = imageToCopy.getImageData();


	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				currentImageData[x+y*xNb+z*xNb*yNb]=imageToCopyData[x+y*xNb+z*xNb*yNb];*/

	////TO DO
}

void Image3D_CPU_half::saveImage(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out | ios::binary);


	if (imageFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " image" << endl;
		half* currentImageData = this->getImageData();
		float* tmp = (float *)malloc(sizeof(float)*this->getDataImageSize());
		halfp2singles(tmp,currentImageData,this->getDataImageSize());
		imageFile.write((char*)tmp,sizeof(float)*this->getDataImageSize());
		imageFile.close();
		delete tmp;
		string name = "chmod 774 ";
		i = system((name + fileName.c_str()).c_str());
		cout << i << "Image saved in " << fileName << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::saveMiddleSliceImage(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out |ios::app | ios::binary);

	if (imageFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " image" << endl;
		half* currentImageData = this->getImageData();
		float* tmp = (float *)malloc(sizeof(float)*this->getXImagePixelNb()*this->getYImagePixelNb());
		halfp2singles(tmp,&currentImageData[this->getZImagePixelNb()/2*this->getXImagePixelNb()*this->getYImagePixelNb()],this->getXImagePixelNb()*this->getYImagePixelNb());
		imageFile.write((char*)tmp,sizeof(float)*this->getXImagePixelNb()*this->getYImagePixelNb());
		imageFile.close();
		delete tmp;
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

void Image3D_CPU_half::loadImage(string fileName)
{
	ifstream imageFile;
	imageFile.open(fileName.c_str(), ios::in|ios::binary);

	if (imageFile.is_open())
	{
		cout << "Loading " << fileName << " image" << endl;

		half* currentImageData = this->getImageData();
		float* tmp = (float *)malloc(sizeof(float)*this->getDataImageSize());
		imageFile.read ((char*)tmp, sizeof(float)*this->getDataImageSize());
		singles2halfp(currentImageData,tmp,this->getDataImageSize());
		delete tmp;
		imageFile.close();
		cout << "Image " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file" << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::setImage(float value)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&value,1);
}

void Image3D_CPU_half::scalarImage(float value)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	half* currentImageData = this->getImageData();

	float tmp;
	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				tmp*=value;
				singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
			}
}

void Image3D_CPU_half::addImage(Image3D_CPU_half* image2)
{
	if(this->isSameSize(image2))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp+=tmp2;
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::addImage(Image3D_CPU_half* image2, float lambda)
{
	if(this->isSameSize(image2))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp=tmp + lambda*tmp2;
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::positiveAddImage(Image3D_CPU_half* image2, float lambda)
{
	if(this->isSameSize(image2))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp=tmp + lambda*tmp2;
					if(tmp<0)
						tmp=0;

					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::diffImage(Image3D_CPU_half* image2)
{
	if(this->isSameSize(image2))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp-=tmp2;
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::diffImage(float lambda, Image3D_CPU_half* image2)
{
	if(this->isSameSize(image2))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp=tmp*lambda - tmp2;
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::multImage(Image3D_CPU_half* image)
{
	if(this->isSameSize(image))
	{

		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* imageData = image->getImageData();
		float tmp,tmp2;

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp,&imageData[x+y*xNb+z*xNb*yNb],1);
					halfp2singles(&tmp2,&currentImageData[x+y*xNb+z*xNb*yNb],1);
					tmp2*=tmp;
					singles2halfp(&currentImageData[x+y*xNb+z*xNb*yNb],&tmp2,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

double Image3D_CPU_half::scalarProductImage(Image3D_CPU_half* image2)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double l1Norm = 0.0;

	half* currentImageData = this->getImageData();
	half* image2Data = image2->getImageData();

	float tmp,tmp2;
	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				halfp2singles(&tmp2,&image2Data[x+y*xNb+z*xNb*yNb],1);
				tmp*=tmp;
				l1Norm+=tmp;
			}

	return l1Norm;
}

double Image3D_CPU_half::getImageL1Norm()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double l1Norm = 0.0;

	half* currentImageData = this->getImageData();
	float tmp;

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				l1Norm+=tmp;
			}

	return l1Norm;
}

double Image3D_CPU_half::getImageL2Norm()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double l2Norm = 0.0;

	half* currentImageData = this->getImageData();
	float tmp;

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				l2Norm+=tmp*tmp;
			}

	return l2Norm;
}

double Image3D_CPU_half::getImageLpNorm(double power)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double lpNorm = 0.0;
	float tmp;

	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				lpNorm += pow(fabs(tmp),power);
			}

	return lpNorm;
}

double Image3D_CPU_half::getImageHuberNorm(double threshold)
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	float tmp = 0;
	double huberNorm=0.0;
	double squareThreshold = threshold*threshold;

	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				if(fabs(tmp) < threshold)
					huberNorm += tmp*tmp;
				else if (fabs(tmp) >= threshold)
					huberNorm += 2.0*threshold*fabs(tmp)-squareThreshold;
				else
					huberNorm += tmp;
			}

	return huberNorm;
}

double Image3D_CPU_half::getImageMean()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageMean = 0.0;
	float tmp;

	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				imageMean += tmp;
			}

	imageMean /= (double)this->getDataImageSize();

	return imageMean;
}

double Image3D_CPU_half::getImageMeanSquare()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageMeanSquare = 0.0;
	float tmp;

	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				imageMeanSquare += tmp*tmp;
			}

	imageMeanSquare /= (double)this->getDataImageSize();

	return imageMeanSquare;
}

double Image3D_CPU_half::getImageStd()
{
	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();
	double imageStd = 0.0;
	double imageMean = this->getImageMean();
	double tmp = 0;
	float tmp1;

	half* currentImageData = this->getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
			{
				halfp2singles(&tmp1,&currentImageData[x+y*xNb+z*xNb*yNb],1);
				tmp = (imageMean - tmp1);
				imageStd += tmp*tmp;
			}

	imageStd /= (double)this->getDataImageSize();

	return sqrt(imageStd);
}

void Image3D_CPU_half::getImageSign(Image3D_CPU_half* signedImage)
{
	if(this->isSameSize(signedImage))
	{
		unsigned long int xn,yn,zn,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* signedImageData = signedImage->getImageData();

		float tmp=0;

		for (zn=0;zn<zNb;zn++)
			for (yn=0;yn<yNb;yn++)
				for (xn=0;xn<xNb;xn++)
				{
					halfp2singles(&tmp,&currentImageData[xn+yn*xNb+zn*xNb*yNb],1);
					tmp = fast_sign(tmp);
					singles2halfp(&signedImageData[xn+yn*xNb+zn*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_CPU_half::getImageAbsPow(Image3D_CPU_half* absPowImage, double p)
{
	if(this->isSameSize(absPowImage))
	{
		unsigned long int xn,yn,zn,xNb,yNb,zNb;
		xNb = this->getXImagePixelNb();
		yNb = this->getYImagePixelNb();
		zNb = this->getZImagePixelNb();

		half* currentImageData = this->getImageData();
		half* absPowImageData = absPowImage->getImageData();

		float tmp=0;

		for (zn=0;zn<zNb;zn++)
			for (yn=0;yn<yNb;yn++)
				for (xn=0;xn<xNb;xn++)
				{
					halfp2singles(&tmp,&currentImageData[xn+yn*xNb+zn*xNb*yNb],1);
					tmp = pow(fabs(tmp), p);
					singles2halfp(&absPowImageData[xn+yn*xNb+zn*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

#include "Image3D_instances.cu"
#include "Image3D_instances_CPU.cu"