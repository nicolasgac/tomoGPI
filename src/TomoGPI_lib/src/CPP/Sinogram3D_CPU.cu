/*
 * Sinogram3D_CPU.cu
 *
  *      Author : gac
 */

#include "Sinogram3D_CPU.cuh"
#include "GPUConstant.cuh"
#include "Sinogram3D_GPU_kernel_half.cuh"
#include "Sinogram3D_GPU_kernel.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
#include "Image3D.cuh"


template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(): Sinogram3D<T>(){}


template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb){}

template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb,T* dataSinogram) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb,dataSinogram){}

template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture,T* dataSinogram) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb,dataSinogram){}

template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb)
{
	T* dataSinogramTmp = this->getDataSinogram();
	this->setCUDAArchitecture(cudaArchitecture);
	std::cout << "\tSino Allocation running on CPU " << sched_getcpu() <<  std::endl;
	checkCudaErrors(cudaHostAlloc ((void **)&dataSinogramTmp,this->getDataSinogramSize()*sizeof(T),cudaHostAllocPortable));
	//dataSinogramTmp = (T*) malloc(sizeof(T)*this->getDataSinogramSize());
	this->setDataSinogram(dataSinogramTmp);
	//	this->setDataSinogram(new T[this->getDataSinogramSize()]());
}


template <typename T>
Sinogram3D_CPU<T>::~Sinogram3D_CPU()
{
	cudaFreeHost(this->getDataSinogram());
	//	delete this->getDataSinogram();
}



template <typename T>
CUDAArchitecture* Sinogram3D_CPU<T>::getCUDAArchitecture() const
{
	return this->cudaArchitecture;
}

template <typename T>
void Sinogram3D_CPU<T>::setCUDAArchitecture(CUDAArchitecture* cudaArchitecture)
{
	this->cudaArchitecture=cudaArchitecture;
}



template <typename T>
Sinogram3D_CPU<T>::Sinogram3D_CPU(const Sinogram3D_CPU& sinogramToCopy)
{
	unsigned long int uNb,vNb,pNb;
	unsigned long long int pixel;

	this->setUSinogramPixelNb(sinogramToCopy.getUSinogramPixelNb());
	this->setVSinogramPixelNb(sinogramToCopy.getVSinogramPixelNb());
	this->setProjectionSinogramNb(sinogramToCopy.getProjectionSinogramNb());

	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	this->setDataSinogramSize(uNb,vNb,pNb);
	this->setDataSinogram(new T[this->getDataSinogramSize()]());

	T* currentSinogramData = this->getDataSinogram();
	T* sinogramToCopyData = sinogramToCopy.getDataSinogram();

#pragma omp parallel for
	for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
		currentSinogramData[pixel]=sinogramToCopyData[pixel];
	}

}


template <typename T>
void Sinogram3D_CPU<T>::setSinogram(T value)
{
	unsigned long int uNb,vNb,pNb;
	unsigned long long int pixel;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();

#pragma omp parallel for
	for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
		currentSinogramData[pixel]=value;
	}
}

template <typename T>
void Sinogram3D_CPU<T>::scalarSinogram(T value)
{
	unsigned long int uNb,vNb,pNb;
	unsigned long long int pixel;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();

#pragma omp parallel for
	for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
		currentSinogramData[pixel]*=value;
	}
}

template <typename T>
void Sinogram3D_CPU<T>::addSinogram(Sinogram3D_CPU<T>* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		unsigned long int uNb,vNb,pNb;
		unsigned long long int pixel;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		T* currentSinogramData = this->getDataSinogram();
		T* sinogramData = sinogram->getDataSinogram();

#pragma omp parallel for
		for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
			currentSinogramData[pixel]+=sinogramData[pixel];
		}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Sinogram3D_CPU<T>::multSinogram(Sinogram3D_CPU<T>* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		unsigned long int uNb,vNb,pNb;
		unsigned long long int pixel;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		T* currentSinogramData = this->getDataSinogram();
		T* sinogramData = sinogram->getDataSinogram();

#pragma omp parallel for
		for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
			currentSinogramData[pixel]*=sinogramData[pixel];
		}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Sinogram3D_CPU<T>::divideSinogram(Sinogram3D_CPU<T>* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		unsigned long int uNb,vNb,pNb;
		unsigned long long int pixel;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		T* currentSinogramData = this->getDataSinogram();
		T* sinogramData = sinogram->getDataSinogram();

#pragma omp parallel for
		for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
			currentSinogramData[pixel]/=sinogramData[pixel];
		}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Sinogram3D_CPU<T>::weightByVariancesNoise(T* v_noise, int stationnary){

	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();

	/* int stationnary :
	 * 1 : one variance per pixel, and same variance at each projection angle
	 * 2 : same variance for all pixels, at all projection angles
	 * 0 or other : one variance per pixel and projection angle (default)
	 */

	for (p=0;p<pNb;p++){
		for (v=0;v<vNb;v++){
			for (u=0;u<uNb;u++){
				if(stationnary==1){
					currentSinogramData[u+v*uNb+p*uNb*vNb]/=v_noise[u+v*uNb];
				}else if(stationnary==2){
					currentSinogramData[u+v*uNb+p*uNb*vNb]/=*(v_noise);
				}else{
					currentSinogramData[u+v*uNb+p*uNb*vNb]/=v_noise[u+v*uNb+p*uNb*vNb];
				}

			}
		}
	}


}

template <typename T>
double Sinogram3D_CPU<T>::sumSinogramWeightedL1(T* v_noise, int stationnary){

	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();

	/* int stationnary :
	 * 1 : one variance per pixel, and same variance at each projection angle
	 * 2 : same variance for all pixels, at all projection angles
	 * 0 or other : one variance per pixel and projection angle (default)
	 */
	double sumSino=0;

	for (p=0;p<pNb;p++){
		for (v=0;v<vNb;v++){
			for (u=0;u<uNb;u++){
				if(stationnary==1){
					sumSino+=double(std::abs(double(currentSinogramData[u+v*uNb+p*uNb*vNb]))/v_noise[u+v*uNb]);
				}else if(stationnary==2){
					sumSino+=double(std::abs(double(currentSinogramData[u+v*uNb+p*uNb*vNb]))/(*(v_noise)));
				}else{
					sumSino+=double(std::abs(double(currentSinogramData[u+v*uNb+p*uNb*vNb]))/v_noise[u+v*uNb+p*uNb*vNb]);
				}

			}
		}
	}

	return sumSino;
}

template <typename T>
double Sinogram3D_CPU<T>::sumSinogramWeightedL2(T* v_noise, int stationnary){

	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	T* currentSinogramData = this->getDataSinogram();

	/* int stationnary :
	 * 1 : one variance per pixel, and same variance at each projection angle
	 * 2 : same variance for all pixels, at all projection angles
	 * 0 or other : one variance per pixel and projection angle (default)
	 */
	double sumSino=0;

	for (p=0;p<pNb;p++){
		for (v=0;v<vNb;v++){
			for (u=0;u<uNb;u++){
				if(stationnary==1){
					sumSino+=double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]/v_noise[u+v*uNb]);
				}else if(stationnary==2){
					sumSino+=double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]/(*(v_noise)));
				}else{
					sumSino+=double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]/v_noise[u+v*uNb+p*uNb*vNb]);
				}

			}
		}
	}

	return sumSino;
}


template <typename T>
void Sinogram3D_CPU<T>::diffSinogram(Sinogram3D_CPU<T>* sinogram1,Sinogram3D_CPU<T>* sinogram2)
{
	if(this->isSameSize(sinogram1) && this->isSameSize(sinogram2))
	{
		unsigned long int uNb,vNb,pNb;
		unsigned long long int pixel;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		T* currentSinogramData = this->getDataSinogram();
		T* sinogram1Data = sinogram1->getDataSinogram();
		T* sinogram2Data = sinogram2->getDataSinogram();

#pragma omp parallel for
		for(pixel=0;pixel<uNb*vNb*pNb;pixel++){
			currentSinogramData[pixel]=sinogram1Data[pixel]-sinogram2Data[pixel];
		}

	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramL1Norm()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double l1Norm = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				l1Norm += double(currentSinogramData[u+v*uNb+p*uNb*vNb]);

	return l1Norm;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramL2Norm()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double l2Norm = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				l2Norm += double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]);

	return l2Norm;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramWeightedL2Norm(T* weights){

	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double l2Norm = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				l2Norm += double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]/weights[u+v*uNb+p*uNb*vNb]);

	return l2Norm;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramLpNorm(double power)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double lpNorm = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				lpNorm += double(pow(fabs(currentSinogramData[u+v*uNb+p*uNb*vNb]),power));

	return lpNorm;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramMean()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramMean = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				sinogramMean += double(currentSinogramData[u+v*uNb+p*uNb*vNb]);

	sinogramMean /= (double)this->getDataSinogramSize();

	return sinogramMean;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramMeanSquare()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramMeanSquare = 0.0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				sinogramMeanSquare += double(currentSinogramData[u+v*uNb+p*uNb*vNb]*currentSinogramData[u+v*uNb+p*uNb*vNb]);



	//cout <<"\t here is L2 norme " << sinogramMeanSquare << endl;
	//cout <<"\t here is Size " << this->getDataSinogramSize() << endl;


	sinogramMeanSquare /= (double)this->getDataSinogramSize();


	return sinogramMeanSquare;
}

template <typename T>
double Sinogram3D_CPU<T>::getSinogramStd()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramStd = 0.0;
	double sinogramMean = this->getSinogramMean();
	double tmp = 0;

	T* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				tmp = double((sinogramMean - currentSinogramData[u+v*uNb+p*uNb*vNb]));
				sinogramStd += tmp*tmp;
			}

	sinogramStd /= (double)this->getDataSinogramSize();

	return sqrt(sinogramStd);
}


/* Sinogram3D_CPU_half definition */

Sinogram3D_CPU_half::Sinogram3D_CPU_half(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture) : Sinogram3D_CPU<half>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb,cudaArchitecture){}

Sinogram3D_CPU_half::~Sinogram3D_CPU_half(){}

Sinogram3D_CPU_half::Sinogram3D_CPU_half(const Sinogram3D_CPU_half& sinogramToCopy)
{
	unsigned long int u,v,p,uNb,vNb,pNb;

	this->setUSinogramPixelNb(sinogramToCopy.getUSinogramPixelNb());
	this->setVSinogramPixelNb(sinogramToCopy.getVSinogramPixelNb());
	this->setProjectionSinogramNb(sinogramToCopy.getProjectionSinogramNb());

	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	this->setDataSinogramSize(uNb,vNb, pNb);
	this->setDataSinogram(new half[this->getDataSinogramSize()]());

	half* currentSinogramData = this->getDataSinogram();
	half* sinogramToCopyData = sinogramToCopy.getDataSinogram();



	float tmp;
	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&sinogramToCopyData[u+v*uNb+p*uNb*vNb],1);
				singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp,1);
			}
}

Sinogram3D_CPU_half & Sinogram3D_CPU_half::operator=(const Sinogram3D_CPU_half &sinogram)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	half* currentSinogramData = this->getDataSinogram();
	half* sinogramData = sinogram.getDataSinogram();

	if((this->getUSinogramPixelNb() == sinogram.getUSinogramPixelNb()) && (this->getVSinogramPixelNb() == sinogram.getVSinogramPixelNb()) && (this->getProjectionSinogramNb() == sinogram.getProjectionSinogramNb()))
	{
		float tmp;
		for (p=0;p<pNb;p++)
			for (v=0;v<vNb;v++)
				for (u=0;u<uNb;u++)
				{
					halfp2singles(&tmp,&sinogramData[u+v*uNb+p*uNb*vNb],1);
					singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp,1);
				}
	}
	else
	{
		cout << "Sinogram must have the same size" << endl;
		exit(EXIT_FAILURE);
	}

	return *this;
}

void Sinogram3D_CPU_half::saveSinogram(string fileName)
{
	ofstream sinogramFile;
	sinogramFile.open(fileName.c_str(),ios::out | ios::binary);


	if (sinogramFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " sinogram" << endl;

		half* currentSinogramData = this->getDataSinogram();
		float* tmp = (float *)malloc(sizeof(float)*this->getDataSinogramSize());
		halfp2singles(tmp,currentSinogramData,this->getDataSinogramSize());
		sinogramFile.write((char*)tmp,sizeof(float)*this->getDataSinogramSize());
		sinogramFile.close();
		delete tmp;
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

void Sinogram3D_CPU_half::loadSinogram(string fileName)
{
	ifstream sinogramFile;
	sinogramFile.open(fileName.c_str(), ios::in|ios::binary);

	if (sinogramFile.is_open())
	{
		cout << "Loading " << fileName << " sinogram" << endl;

		half* currentSinogramData = this->getDataSinogram();
		float* tmp = (float *)malloc(sizeof(float)*this->getDataSinogramSize());
		sinogramFile.read ((char*)tmp, sizeof(float)*this->getDataSinogramSize());
		singles2halfp(currentSinogramData,tmp,this->getDataSinogramSize());

		sinogramFile.close();

		cout << "Sinogram " << fileName << " loaded" << endl;
	}
	else
	{
		cout << "Unable to open file" << fileName << endl;
		exit(EXIT_FAILURE);
	}
}

void Sinogram3D_CPU_half::setSinogram(float value)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	half* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&value,1);
}

void Sinogram3D_CPU_half::scalarSinogram(float value)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	half* currentSinogramData = this->getDataSinogram();
	float tmp;

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++){
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				tmp*=value;
				singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp,1);
			}
}

void Sinogram3D_CPU_half::addSinogram(Sinogram3D_CPU_half* sinogram)
{
	if(this->isSameSize(sinogram))
	{

		unsigned long int u,v,p,uNb,vNb,pNb;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		half* currentSinogramData = this->getDataSinogram();
		half* sinogramData = sinogram->getDataSinogram();
		float tmp,tmp2;

		for (p=0;p<pNb;p++)
			for (v=0;v<vNb;v++)
				for (u=0;u<uNb;u++)
				{
					halfp2singles(&tmp,&sinogramData[u+v*uNb+p*uNb*vNb],1);
					halfp2singles(&tmp2,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
					tmp2+=tmp;
					singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp2,1);
				}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Sinogram3D_CPU_half::multSinogram(Sinogram3D_CPU_half* sinogram)
{
	if(this->isSameSize(sinogram))
	{

		unsigned long int u,v,p,uNb,vNb,pNb;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		half* currentSinogramData = this->getDataSinogram();
		half* sinogramData = sinogram->getDataSinogram();
		float tmp,tmp2;

		for (p=0;p<pNb;p++)
			for (v=0;v<vNb;v++)
				for (u=0;u<uNb;u++)
				{
					halfp2singles(&tmp,&sinogramData[u+v*uNb+p*uNb*vNb],1);
					halfp2singles(&tmp2,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
					tmp2*=tmp;
					singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp2,1);
				}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Sinogram3D_CPU_half::diffSinogram(Sinogram3D_CPU_half* sinogram1,Sinogram3D_CPU_half* sinogram2)
{
	if(this->isSameSize(sinogram1) && this->isSameSize(sinogram2))
	{

		unsigned long int u,v,p,uNb,vNb,pNb;
		uNb = this->getUSinogramPixelNb();
		vNb = this->getVSinogramPixelNb();
		pNb = this->getProjectionSinogramNb();

		half* currentSinogramData = this->getDataSinogram();
		half* sinogram1Data = sinogram1->getDataSinogram();
		half* sinogram2Data = sinogram2->getDataSinogram();
		float tmp,tmp1,tmp2;

		for (p=0;p<pNb;p++)
			for (v=0;v<vNb;v++)
				for (u=0;u<uNb;u++)
				{
					halfp2singles(&tmp,&sinogram1Data[u+v*uNb+p*uNb*vNb],1);
					halfp2singles(&tmp1,&sinogram2Data[u+v*uNb+p*uNb*vNb],1);
					halfp2singles(&tmp2,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
					tmp2=tmp-tmp1;
					singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp2,1);
				}
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

double Sinogram3D_CPU_half::getSinogramL1Norm()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double l1Norm = 0.0;

	half* currentSinogramData = this->getDataSinogram();
	float tmp;

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				l1Norm+=tmp;
			}

	return l1Norm;
}

double Sinogram3D_CPU_half::getSinogramL2Norm()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double l2Norm = 0.0;

	half* currentSinogramData = this->getDataSinogram();
	float tmp;

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				l2Norm+=tmp*tmp;
			}

	return l2Norm;
}

double Sinogram3D_CPU_half::getSinogramLpNorm(double power)
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double lpNorm = 0.0;
	float tmp;

	half* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				lpNorm += pow(fabs(tmp),power);
			}

	return lpNorm;
}

double Sinogram3D_CPU_half::getSinogramMean()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramMean = 0.0;
	float tmp;

	half* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				sinogramMean += tmp;
			}

	sinogramMean /= (double)this->getDataSinogramSize();

	return sinogramMean;
}

double Sinogram3D_CPU_half::getSinogramMeanSquare()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramMeanSquare = 0.0;
	float tmp;

	half* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				sinogramMeanSquare += tmp*tmp;
			}

	sinogramMeanSquare /= (double)this->getDataSinogramSize();

	return sinogramMeanSquare;
}

double Sinogram3D_CPU_half::getSinogramStd()
{
	unsigned long int u,v,p,uNb,vNb,pNb;
	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();
	double sinogramStd = 0.0;
	double sinogramMean = this->getSinogramMean();
	double tmp = 0;
	float tmp1;

	half* currentSinogramData = this->getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp1,&currentSinogramData[u+v*uNb+p*uNb*vNb],1);
				tmp = (sinogramMean - tmp1);
				sinogramStd += tmp*tmp;
			}

	sinogramStd /= (double)this->getDataSinogramSize();

	return sqrt(sinogramStd);
}

#include "Sinogram3D_instances.cu"
#include "Sinogram3D_instances_CPU.cu"