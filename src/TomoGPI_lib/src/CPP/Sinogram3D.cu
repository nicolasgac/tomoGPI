/*
 * Sinogram3D.cu
 *
 *      Author: gac
 */

#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"

#include "GPUConstant.cuh"
#include "Sinogram3D_GPU_kernel_half.cuh"
#include "Sinogram3D_GPU_kernel.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
#include "Image3D.cuh"

template <typename T>
Sinogram3D<T>::Sinogram3D() : uSinogramPixelNb(0), vSinogramPixelNb(0), projectionSinogramNb(0), dataSinogramSize(0), dataSinogram(0){}

template <typename T>
Sinogram3D<T>::Sinogram3D(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb,T* dataSino) : uSinogramPixelNb(uSinogramPixelNb), vSinogramPixelNb(vSinogramPixelNb), projectionSinogramNb(projectionSinogramNb), dataSinogram(dataSino)
{
	this->dataSinogramSize = uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
}

template <typename T>
Sinogram3D<T>::Sinogram3D(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb) : uSinogramPixelNb(uSinogramPixelNb), vSinogramPixelNb(vSinogramPixelNb), projectionSinogramNb(projectionSinogramNb), dataSinogram(0)
{
	this->dataSinogramSize = uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
	cout <<"======= New Sinogram Object is Created   ===   Size in bytes is : "<<sizeof(T)*(this->dataSinogramSize)<<" ======="<<endl;
}

template <typename T>
Sinogram3D<T>::~Sinogram3D(){}

template <typename T>
Sinogram3D<T> & Sinogram3D<T>::operator=(const Sinogram3D<T> &sinogram)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();
	T* sinogramData = sinogram.getDataSinogram();

	if((this->getUSinogramPixelNb() == sinogram.getUSinogramPixelNb()) && (this->getVSinogramPixelNb() == sinogram.getVSinogramPixelNb()) && (this->getProjectionSinogramNb() == sinogram.getProjectionSinogramNb()))
	{
		for (p=0;p<pNb;p++)
			for (v=0;v<vNb;v++)
				for (u=0;u<uNb;u++)
					currentSinogramData[u+v*uNb+p*uNb*vNb]=sinogramData[u+v*uNb+p*uNb*vNb];
	}
	else
	{
		cout << "Sinogram must have the same size" << endl;
		exit(EXIT_FAILURE);
	}

	return *this;
}

template <typename T>
bool Sinogram3D<T>::isSameSize(Sinogram3D<T>* sinogram2) const
{
	return (this->getUSinogramPixelNb() == sinogram2->getUSinogramPixelNb()) && (this->getVSinogramPixelNb() == sinogram2->getVSinogramPixelNb()) && (this->getProjectionSinogramNb() == sinogram2->getProjectionSinogramNb());
}

template <typename T>
unsigned long int Sinogram3D<T>::getUSinogramPixelNb() const
{
	return uSinogramPixelNb;
}

template <typename T>
unsigned long int Sinogram3D<T>::getVSinogramPixelNb() const
{
	return vSinogramPixelNb;
}

template <typename T>
unsigned long int Sinogram3D<T>::getProjectionSinogramNb() const
{
	return projectionSinogramNb;
}

template <typename T>
T* Sinogram3D<T>::getDataSinogram() const
{
	return dataSinogram;
}

template <typename T>
unsigned long int Sinogram3D<T>::getDataSinogramSize() const
{
	return dataSinogramSize;
}
 
template <typename T>
void Sinogram3D<T>::setUSinogramPixelNb(unsigned long int uSinogramPixelNb)
{
	this->uSinogramPixelNb = uSinogramPixelNb;
	setDataSinogramSize(this->uSinogramPixelNb,this->vSinogramPixelNb,this->projectionSinogramNb);
}

template <typename T>
void Sinogram3D<T>::setVSinogramPixelNb(unsigned long int vSinogramPixelNb)
{
	this->vSinogramPixelNb = uSinogramPixelNb;
	setDataSinogramSize(this->uSinogramPixelNb,this->vSinogramPixelNb,this->projectionSinogramNb);
}

template <typename T>
void Sinogram3D<T>::setProjectionSinogramNb(unsigned long int projectionSinogramNb)
{
	this->projectionSinogramNb = projectionSinogramNb;
	setDataSinogramSize(this->uSinogramPixelNb,this->vSinogramPixelNb,this->projectionSinogramNb);
}

template <typename T>
void Sinogram3D<T>::setDataSinogram(T* dataSinogram)
{
	this->dataSinogram = dataSinogram;
}

template <typename T>
void Sinogram3D<T>::setDataSinogramSize(unsigned long int uSinogramPixelNb,unsigned long int vSinogramPixelNb,unsigned long int projectionSinogramNb)
{
	this->uSinogramPixelNb = uSinogramPixelNb;
	this->vSinogramPixelNb = vSinogramPixelNb;
	this->projectionSinogramNb = projectionSinogramNb;
	this->dataSinogramSize = this->uSinogramPixelNb*this->vSinogramPixelNb*this->projectionSinogramNb;
}

template <typename T>
void Sinogram3D<T>::saveSinogram(string fileName)
{
	ofstream sinogramFile;
	sinogramFile.open(fileName.c_str(),ios::out | ios::binary);

	if (sinogramFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " sinogram" << endl;
		sinogramFile.write((char*)this->dataSinogram,sizeof(T)*this->dataSinogramSize);
		sinogramFile.close();
		string name = "chmod 774 ";
		i=system((name + fileName.c_str()).c_str());
		cout << i << "Sinogram saved in " << fileName << endl;
	}
	else
	{
		cout << "Unable to open file " << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Sinogram3D<T>::loadSinogram(string fileName)
{
	ifstream sinogramFile;
	sinogramFile.open(fileName.c_str(), ios::in|ios::binary);

	if (sinogramFile.is_open())
	{
		cout << "Loading " << fileName << " sinogram" << endl;
		cout << "size : "<< sizeof(T)*this->getDataSinogramSize() << endl;
		//float* tmp = (float *)malloc(sizeof(float)*this->getDataSinogramSize());
		//cout << "ptr :" << tmp << endl;
		//sinogramFile.read ((char*)tmp, sizeof(T)*this->dataSinogramSize);
		cout << "ptr :" << this->getDataSinogram() << endl;
		sinogramFile.read ((char*)this->dataSinogram, sizeof(T)*this->dataSinogramSize);
		sinogramFile.close(); 

		cout << "Sinogram " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file" << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

template < typename T>
void Sinogram3D<T>::loadSinogram_InitIter(string fileName)
{
	ifstream sinogramFile;
	sinogramFile.open(fileName.c_str(), ios::in|ios::binary);

	if (sinogramFile.is_open())
	{
		cout << "Loading " << fileName << " sinogram" << endl;
		sinogramFile.read ((char*)this->dataSinogram, sizeof(T)*this->dataSinogramSize);
		sinogramFile.close();

		cout << "Sinogram " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file" << fileName << endl;
		exit(EXIT_FAILURE);
	}
}
/*
template<typename T>
char Sinogram3D<T>::getFdk()
{
	return this->fdk;
}

template<typename T>
void Sinogram3D<T>::setFdk(char fdk)
{
	this->fdk = fdk;
}*/

/*
template < typename T>
void Sinogram3D<T>::InitSinogram3D(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk){}

template < typename T>
void Sinogram3D<T>::InitSinoBack(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk){}

template < typename T>
void Sinogram3D<T>::InitSinogram3D_InitSG(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk){}

template < typename T>
void Sinogram3D<T>::InitSinogram3D_v4(Acquisition* acquisition, Detector* detector,CUDABProjectionArchitecture* cudabackprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram, char fdk){}

template < typename T>
void Sinogram3D<T>::MGPUCopy(){}

template < typename T>
void Sinogram3D<T>::GPUCopy_Simple(Volume<T>* volume, Sinogram3D<T>* sinogram){}
*/

/*
template<typename T>
Acquisition* Sinogram3D<T>::getAcquisition() const
{
	return this->acquisition;
}

template<typename T>
Detector* Sinogram3D<T>::getDetector() const
{
	return this->detector;
}

template<typename T>
void Sinogram3D<T>::setAcquisition(Acquisition* acquisition)
{
	this->acquisition = acquisition;
}

template<typename T>
void Sinogram3D<T>::setDetector(Detector* detector)
{
	this->detector = detector;
}
*/

#include "Sinogram3D_instances.cu"