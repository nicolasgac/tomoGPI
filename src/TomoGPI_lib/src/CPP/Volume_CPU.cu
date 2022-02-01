/*
 * Volume_CPU.cu
 *
 *      Author: gac
 */

#include "Volume_CPU.cuh"
#include "GPUConstant.cuh"

/* Volume_CPU definition */
template <typename T>
Volume_CPU<T>::Volume_CPU() : Volume<T>(){}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,dataImage));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}


template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture, T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture,dataImage));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::~Volume_CPU()
{
	delete this->getVolumeImage();
}

/*template <typename T>
Volume_CPU<T>&  Volume_CPU<T>::operator=(const Volume_CPU<T> &volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
		this->setYVolumeSize(volumeToCopy.getYVolumeSize());
		this->setZVolumeSize(volumeToCopy.getZVolumeSize());

		this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
		this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
		this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

		this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
		this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
		this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

		this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
		this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
		this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

		this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
		this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
		this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

		this->getVolumeImage()->copyImage3D(volumeToCopy.getVolumeImage());


	return *this;
}*/

template <typename T>
Volume_CPU<T>::Volume_CPU(const Volume_CPU<T>& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_CPU<T>(*volumeToCopy.getVolumeImage()));
}

template <typename T>
Image3D_CPU<T>* Volume_CPU<T>::getVolumeImage() const
{
	return (Image3D_CPU<T>*)Volume<T>::getVolumeImage();
}

template <typename T>
void Volume_CPU<T>::setVolume(T value)
{
	this->getVolumeImage()->setImage(value);
}

template <typename T>
void Volume_CPU<T>::scalarVolume(T value)
{
	this->getVolumeImage()->scalarImage(value);
}

template <typename T>
void Volume_CPU<T>::addVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::addVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_CPU<T>::positiveAddVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_CPU<T>::diffVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::diffVolume(T lambda, Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::diffVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage(), lambda);
}

template <typename T>
void Volume_CPU<T>::multVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

template <typename T>
double Volume_CPU<T>::scalarProductVolume(Volume_CPU<T>* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

template <typename T>
double Volume_CPU<T>::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

template <typename T>
double Volume_CPU<T>::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

template <typename T>
double Volume_CPU<T>::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

template <typename T>
double Volume_CPU<T>::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm(threshold);
}

template <typename T>
double Volume_CPU<T>::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

template <typename T>
double Volume_CPU<T>::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

template <typename T>
double Volume_CPU<T>::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

template <typename T>
void Volume_CPU<T>::getVolumeSign(Volume_CPU<T>* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::getVolumeAbsPow(Volume_CPU<T>* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}




template <typename T>
void Volume_CPU<T>::grad_xplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(xn==(Nx-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_xmoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(xn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[(xn-1)+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_yplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(yn==(Ny-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_ymoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(yn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[xn+(yn-1)*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_zplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(zn==(Nz-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_zmoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(zn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+(zn-1)*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::sign_volume(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* sign_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data
	T value;
	int signe_volume;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				value=data_volume[xn+yn*Nx+zn*Nx*Ny];
				if(value>0){
					signe_volume=1;
				}else if(value<0){
					signe_volume=-1;
				}else{
					signe_volume=0;
				}
				sign_data[xn+yn*Nx+zn*Nx*Ny]=T(signe_volume);
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::weightVolume(T* weights){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* data_volume=this->getVolumeData();

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				data_volume[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]/weights[xn+yn*Nx+zn*Nx*Ny];
			}
		}
	}

}

template <typename T>
double Volume_CPU<T>::sumWeightedVolume(T* weights){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* data_volume=this->getVolumeData();
	double sum_weighted=0.0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				sum_weighted+=(double(data_volume[xn+yn*Nx+zn*Nx*Ny])*double(data_volume[xn+yn*Nx+zn*Nx*Ny])/double(weights[xn+yn*Nx+zn*Nx*Ny]));
			}
		}
	}
	return sum_weighted;
}

#ifdef __CUDACC__
template <typename T>
__host__ void Volume_CPU<T>::copyConstantGPU()
{
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();

	cudaMemcpyToSymbol(xVolumePixelNb_GPU,&Nx , sizeof(unsigned long int));
	cudaMemcpyToSymbol(yVolumePixelNb_GPU,&Ny , sizeof(unsigned long int));
	cudaMemcpyToSymbol(zVolumePixelNb_GPU,&Nz , sizeof(unsigned long int));

	std::cout << "Lancement mémoire constante \n" << std::endl;
}
#endif

/* Volume_CPU_half definition */

Volume_CPU_half::Volume_CPU_half() : Volume_CPU<half>(){}

Volume_CPU_half::Volume_CPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) //: Volume<half>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setXVolumeSize(xVolumeSize);
	this->setYVolumeSize(yVolumeSize);
	this->setZVolumeSize(zVolumeSize);
	this->setXVolumePixelNb(xVolumePixelNb);
	this->setYVolumePixelNb(yVolumePixelNb);
	this->setZVolumePixelNb(zVolumePixelNb);
	this->setXVolumePixelSize(xVolumeSize/xVolumePixelNb);
	this->setYVolumePixelSize(yVolumeSize/yVolumePixelNb);
	this->setZVolumePixelSize(this->getXVolumePixelSize());
	this->setXVolumeCenterPixel(xVolumePixelNb/2.0 - 0.5);
	this->setYVolumeCenterPixel(yVolumePixelNb/2.0 - 0.5);
	this->setZVolumeCenterPixel(zVolumePixelNb/2.0 - 0.5);
	this->setVolumeImage(new Image3D_CPU_half(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

Volume_CPU_half::~Volume_CPU_half()
{
//		delete (Image3D_CPU_half*)this->getVolumeImage();
}


Volume_CPU_half::Volume_CPU_half(const Volume_CPU_half& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_CPU_half(*volumeToCopy.getVolumeImage()));
}

Image3D_CPU_half* Volume_CPU_half::getVolumeImage() const
{
	return (Image3D_CPU_half*)Volume_CPU<half>::getVolumeImage();
}

void Volume_CPU_half::setVolume(float value)
{
	this->getVolumeImage()->setImage(value);
}

void Volume_CPU_half::scalarVolume(float value)
{
	this->getVolumeImage()->scalarImage(value);
}

void Volume_CPU_half::addVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

void Volume_CPU_half::addVolume(Volume_CPU_half* volume2, float lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

void Volume_CPU_half::positiveAddVolume(Volume_CPU_half* volume2, float lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

void Volume_CPU_half::diffVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

void Volume_CPU_half::diffVolume(float lambda, Volume_CPU_half* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

void Volume_CPU_half::multVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

double Volume_CPU_half::scalarProductVolume(Volume_CPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

double Volume_CPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

double Volume_CPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

double Volume_CPU_half::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

double Volume_CPU_half::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm(threshold);
}

double Volume_CPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

double Volume_CPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

double Volume_CPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

void Volume_CPU_half::getVolumeSign(Volume_CPU_half* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

void Volume_CPU_half::getVolumeAbsPow(Volume_CPU_half* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}

void Volume_CPU_half::saveVolume(string fileName)
{
	this->getVolumeImage()->saveImage(fileName);
}

void Volume_CPU_half::saveMiddleSliceVolume(string fileName)
{
	this->getVolumeImage()->saveMiddleSliceImage(fileName);
}

void Volume_CPU_half::loadVolume(string fileName)
{
	this->getVolumeImage()->loadImage(fileName);
}





void Volume_CPU_half::grad_xplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_xmoins(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_yplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_ymoins(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_zplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_zmoins(Volume_CPU_half* volume){}

void Volume_CPU_half::sign_volume(Volume_CPU_half* volume){}

#include "Volume_instances.cu"
#include "Volume_instances_CPU.cu"
