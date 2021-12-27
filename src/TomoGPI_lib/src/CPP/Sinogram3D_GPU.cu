/*
 * Sinogram3D_GPU.cu
 *
 *      Author: gac
 */


#include "Sinogram3D_GPU.cuh"
#include "GPUConstant.cuh"
#include "Sinogram3D_GPU_kernel_half.cuh"
#include "Sinogram3D_GPU_kernel.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
#include "Image3D.cuh"


/* Sinogram3D_GPU definition */

template <typename T>
Sinogram3D_GPU<T>::Sinogram3D_GPU() : Sinogram3D<T>(), cudaArchitecture(0){}

template <typename T>
Sinogram3D_GPU<T>::Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb){}

template <typename T>
Sinogram3D_GPU<T>::Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture,T* dataSinogram) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb,dataSinogram), cudaArchitecture(cudaArchitecture){}

template <typename T>
Sinogram3D_GPU<T>::Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture) : Sinogram3D<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb), cudaArchitecture(cudaArchitecture)
{
	T* dataSinogramTmp = this->getDataSinogram();
	checkCudaErrors(cudaMallocManaged ((void **)&dataSinogramTmp, sizeof(T)*this->getDataSinogramSize(),cudaMemAttachGlobal));
	checkCudaErrors(cudaMemset(dataSinogramTmp,0.0,sizeof(T)*this->getDataSinogramSize()));
	this->setDataSinogram(dataSinogramTmp);
	this->copyConstantGPU();
}

template <typename T>
Sinogram3D_GPU<T>::~Sinogram3D_GPU()
{
	T* dataSinogramTmp = this->getDataSinogram();
	checkCudaErrors(cudaFree (dataSinogramTmp));
}

template <typename T>
Sinogram3D_GPU<T>::Sinogram3D_GPU(const Sinogram3D_GPU& sinogramToCopy)
{
	unsigned long int u,v,p,uNb,vNb,pNb;

	this->setUSinogramPixelNb(sinogramToCopy.getUSinogramPixelNb());
	this->setVSinogramPixelNb(sinogramToCopy.getVSinogramPixelNb());
	this->setProjectionSinogramNb(sinogramToCopy.getProjectionSinogramNb());

	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	this->setDataSinogramSize(uNb,vNb,pNb);

	this->cudaArchitecture = new CUDAArchitecture();
	*this->cudaArchitecture = *sinogramToCopy.cudaArchitecture;

	T* currentSinogramData = this->getDataSinogram();
	checkCudaErrors(cudaMallocManaged ((void **)&currentSinogramData, sizeof(T)*this->getDataSinogramSize(),cudaMemAttachGlobal));
	cudaMemset(currentSinogramData,0.0,sizeof(T)*this->getDataSinogramSize());
	this->setDataSinogram(currentSinogramData);
	currentSinogramData = this->getDataSinogram();
	T* sinogramToCopyData = sinogramToCopy.getDataSinogram();

	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
				currentSinogramData[u+v*uNb+p*uNb*vNb]=sinogramToCopyData[u+v*uNb+p*uNb*vNb];
}

template <typename T>
void Sinogram3D_GPU<T>::saveSinogram(string fileName)
{
	ofstream sinogramFile;
	sinogramFile.open(fileName.c_str(),ios::out | ios::binary);


	if (sinogramFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " sinogram GPU to CPU" << endl;

		T* currentSinogramData = this->getDataSinogram();
		T* currentSinogramData_CPU ;
		currentSinogramData_CPU=(T*)malloc(sizeof(T)*this->getDataSinogramSize());;

		checkCudaErrors(cudaMemcpy(currentSinogramData_CPU,this->getDataSinogram(),this->getDataSinogramSize()*sizeof(T),cudaMemcpyDeviceToHost));


		sinogramFile.write((char *)currentSinogramData_CPU,sizeof(float)*this->getDataSinogramSize());
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
CUDAArchitecture* Sinogram3D_GPU<T>::getCUDAArchitecture() const
{
	return this->cudaArchitecture;
}

template <typename T>
void Sinogram3D_GPU<T>::setCUDAArchitecture(CUDAArchitecture* cudaArchitecture)
{
	this->cudaArchitecture=cudaArchitecture;
}

template <typename T>
void Sinogram3D_GPU<T>::setSinogram(T value)
{
	T* currentSinogramData = this->getDataSinogram();

	dim3 dimBlock(this->cudaArchitecture->getXThreadNb(), this->cudaArchitecture->getYThreadNb(), this->cudaArchitecture->getZThreadNb());
	dim3 dimGrid(this->cudaArchitecture->getXBlockNb(), this->cudaArchitecture->getYBlockNb(), this->cudaArchitecture->getZBlockNb());

	setSinogram_k<T><<<dimGrid,dimBlock>>>(currentSinogramData, value, this->getDataSinogramSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
void Sinogram3D_GPU<T>::scalarSinogram(T value)
{
	T* currentSinogramData = this->getDataSinogram();

	dim3 dimBlock(this->cudaArchitecture->getXThreadNb(), this->cudaArchitecture->getYThreadNb(), this->cudaArchitecture->getZThreadNb());
	dim3 dimGrid(this->cudaArchitecture->getXBlockNb(), this->cudaArchitecture->getYBlockNb(), this->cudaArchitecture->getZBlockNb());

	scalarSinogram_k<T><<<dimGrid,dimBlock>>>(currentSinogramData, value, this->getDataSinogramSize());
	checkCudaErrors(cudaThreadSynchronize());
}

template <typename T>
void Sinogram3D_GPU<T>::addSinogram(Sinogram3D_GPU<T>* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		T* currentSinogramData = this->getDataSinogram();
		T* sinogramData = sinogram->getDataSinogram();

		dim3 dimBlock(this->cudaArchitecture->getXThreadNb(), this->cudaArchitecture->getYThreadNb(), this->cudaArchitecture->getZThreadNb());
		dim3 dimGrid(this->cudaArchitecture->getXBlockNb(), this->cudaArchitecture->getYBlockNb(), this->cudaArchitecture->getZBlockNb());

		addSinogram_k<T><<<dimGrid,dimBlock>>>(currentSinogramData,sinogramData,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaThreadSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Sinogram3D_GPU<T>::multSinogram(Sinogram3D_GPU<T>* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		T* currentSinogramData = this->getDataSinogram();
		T* sinogramData = sinogram->getDataSinogram();

		dim3 dimBlock(this->cudaArchitecture->getXThreadNb(), this->cudaArchitecture->getYThreadNb(), this->cudaArchitecture->getZThreadNb());
		dim3 dimGrid(this->cudaArchitecture->getXBlockNb(), this->cudaArchitecture->getYBlockNb(), this->cudaArchitecture->getZBlockNb());

		multSinogram_k<T><<<dimGrid,dimBlock>>>(currentSinogramData,sinogramData,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

// weight current sinogram
template <typename T>
void Sinogram3D_GPU<T>::weightByVariancesNoise(T* v_noise, int stationnary)
{

}

// sum of squared weighted current sinogram
template <typename T>
double Sinogram3D_GPU<T>::sumSinogramWeightedL1(T* v_noise, int stationnary)
{
	return 0;
}

// sum of squared weighted current sinogram
template <typename T>
double Sinogram3D_GPU<T>::sumSinogramWeightedL2(T* v_noise, int stationnary)
{
	return 0;
}

template <typename T>
void Sinogram3D_GPU<T>::diffSinogram(Sinogram3D_GPU<T>* sinogram1,Sinogram3D_GPU<T>* sinogram2)
{
	if(this->isSameSize(sinogram1) && this->isSameSize(sinogram2))
	{
		T* currentSinogramData = this->getDataSinogram();
		T* sinogram1Data = sinogram1->getDataSinogram();
		T* sinogram2Data = sinogram2->getDataSinogram();

		dim3 dimBlock(this->cudaArchitecture->getXThreadNb(), this->cudaArchitecture->getYThreadNb(), this->cudaArchitecture->getZThreadNb());
		dim3 dimGrid(this->cudaArchitecture->getXBlockNb(), this->cudaArchitecture->getYBlockNb(), this->cudaArchitecture->getZBlockNb());



		diffSinogram_k<T><<<dimGrid,dimBlock>>>(sinogram1Data,sinogram2Data,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
T Sinogram3D_GPU<T>::getSinogramL1Norm()
{
	T* currentSinogramData = this->getDataSinogram();
	T* g_odata;
	T l1Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(T),cudaMemAttachGlobal));
	sinogramReduce<T,256,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

template <typename T>
T Sinogram3D_GPU<T>::getSinogramL2Norm()
{
	T* currentSinogramData = this->getDataSinogram();
	T* g_odata;
	T l2Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	cout << "threads " << threads << "blocks " << blocks << " smemSize " <<  smemSize <<endl;
	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(T),cudaMemAttachGlobal));
	sinogramReduce_square<T,256,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += g_odata[i];

	return l2Norm;
}

template <typename T>
T Sinogram3D_GPU<T>::getSinogramLpNorm(T power)
{
	T* currentSinogramData = this->getDataSinogram();
	T* g_odata;
	T lpNorm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(T),cudaMemAttachGlobal));
	sinogramReduce_abspow<T,256,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];

	return lpNorm;
}

template <typename T>
T Sinogram3D_GPU<T>::getSinogramMean()
{
	T sinogramMean = this->getSinogramL1Norm();
	sinogramMean/=(T)this->getDataSinogramSize();

	return sinogramMean;
}

template <typename T>
T Sinogram3D_GPU<T>::getSinogramMeanSquare()
{
	T sinogramMeanSquare = this->getSinogramL2Norm();
	//cout <<"\t here is L2 norme " << sinogramMeanSquare << endl;


	sinogramMeanSquare/=(T)this->getDataSinogramSize();

	//cout <<"\t here is Size " << this->getDataSinogramSize() << endl;

	return sinogramMeanSquare;
}


template <typename T>
T Sinogram3D_GPU<T>::getSinogramStd()
{
	T sinogramMean = this->getSinogramMean();
	T* currentSinogramData = this->getDataSinogram();
	T* g_odata;
	T sinogramStd=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(T),cudaMemAttachGlobal));
	sinogramReduce_std<T,256,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, sinogramMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		sinogramStd += g_odata[i];

	sinogramStd/=(T)this->getDataSinogramSize();

	return sqrt(sinogramStd);
}


/* Copy Sinogram_GPU constant */
template <typename T>
__host__ void Sinogram3D_GPU<T>::copyConstantGPU()
{
	unsigned long int uSinogramPixelNb = this->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = this->getVSinogramPixelNb();

	cudaMemcpyToSymbol(uSinogramPixelNb_GPU,&uSinogramPixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(vSinogramPixelNb_GPU,&vSinogramPixelNb,sizeof(unsigned long int));
}


/* Sinogram3D_GPU_half definition */
Sinogram3D_GPU_half::Sinogram3D_GPU_half(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture) : Sinogram3D_GPU<half>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb, cudaArchitecture){}

Sinogram3D_GPU_half::~Sinogram3D_GPU_half(){}

Sinogram3D_GPU_half::Sinogram3D_GPU_half(const Sinogram3D_GPU_half& sinogramToCopy)
{
	unsigned long int u,v,p,uNb,vNb,pNb;

	this->setUSinogramPixelNb(sinogramToCopy.getUSinogramPixelNb());
	this->setVSinogramPixelNb(sinogramToCopy.getVSinogramPixelNb());
	this->setProjectionSinogramNb(sinogramToCopy.getProjectionSinogramNb());

	uNb = this->getUSinogramPixelNb();
	vNb = this->getVSinogramPixelNb();
	pNb = this->getProjectionSinogramNb();

	this->setDataSinogramSize(uNb,vNb, pNb);

	half* currentSinogramData = this->getDataSinogram();
	half* sinogramToCopyData = sinogramToCopy.getDataSinogram();

	checkCudaErrors(cudaMallocManaged ((void **)&currentSinogramData, sizeof(half)*this->getDataSinogramSize(),cudaMemAttachGlobal));
	cudaMemset(currentSinogramData,0.0,sizeof(half)*this->getDataSinogramSize());

	float tmp;
	for (p=0;p<pNb;p++)
		for (v=0;v<vNb;v++)
			for (u=0;u<uNb;u++)
			{
				halfp2singles(&tmp,&sinogramToCopyData[u+v*uNb+p*uNb*vNb],1);
				singles2halfp(&currentSinogramData[u+v*uNb+p*uNb*vNb],&tmp,1);
			}
}

Sinogram3D_GPU_half & Sinogram3D_GPU_half::operator=(const Sinogram3D_GPU_half &sinogram)
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

void Sinogram3D_GPU_half::saveSinogram(string fileName)
{
	ofstream sinogramFile;
	sinogramFile.open(fileName.c_str(),ios::out | ios::binary);


	if (sinogramFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " sinogram with conversion from half to float" << endl;

		half* currentSinogramData = this->getDataSinogram();
		float* tmp = (float *)malloc(sizeof(float)*this->getDataSinogramSize());
		Sinogram3D_GPU<float>* sino_32f = new Sinogram3D_GPU<float>(this->getUSinogramPixelNb(), this->getVSinogramPixelNb(), this->getProjectionSinogramNb(),this->getCUDAArchitecture());


		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		half_to_float_kernel<<<dimGrid,dimBlock>>>(currentSinogramData,sino_32f->getDataSinogram());
		checkCudaErrors(cudaDeviceSynchronize());




		sinogramFile.write((char *)sino_32f->getDataSinogram(),sizeof(float)*this->getDataSinogramSize());
		sinogramFile.close();
		delete sino_32f;
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

void Sinogram3D_GPU_half::loadSinogram(string fileName)
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

void Sinogram3D_GPU_half::setSinogram(float value)
{
	half* currentSinogramData = this->getDataSinogram();

	dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
	dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

	setSinogram_k_half<half><<<dimGrid,dimBlock>>>(currentSinogramData, value, this->getDataSinogramSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

void Sinogram3D_GPU_half::scalarSinogram(float value)
{
	half* currentSinogramData = this->getDataSinogram();

	dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
	dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

	scalarSinogram_k_half<half><<<dimGrid,dimBlock>>>(currentSinogramData, value, this->getDataSinogramSize());
	checkCudaErrors(cudaThreadSynchronize());
}

void Sinogram3D_GPU_half::addSinogram(Sinogram3D_GPU_half* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		half* currentSinogramData = this->getDataSinogram();
		half* sinogramData = sinogram->getDataSinogram();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		addSinogram_k_half<half><<<dimGrid,dimBlock>>>(currentSinogramData,sinogramData,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaThreadSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Sinogram3D_GPU_half::multSinogram(Sinogram3D_GPU_half* sinogram)
{
	if(this->isSameSize(sinogram))
	{
		half* currentSinogramData = this->getDataSinogram();
		half* sinogramData = sinogram->getDataSinogram();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		multSinogram_k_half<half><<<dimGrid,dimBlock>>>(currentSinogramData,sinogramData,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Sinogram3D_GPU_half::diffSinogram(Sinogram3D_GPU_half* sinogram1,Sinogram3D_GPU_half* sinogram2)
{
	if(this->isSameSize(sinogram1) && this->isSameSize(sinogram2))
	{
		half* currentSinogramData = this->getDataSinogram();
		half* sinogram1Data = sinogram1->getDataSinogram();
		half* sinogram2Data = sinogram2->getDataSinogram();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		diffSinogram_k_half<half><<<dimGrid,dimBlock>>>(sinogram1Data,sinogram2Data,currentSinogramData, this->getDataSinogramSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Sinograms must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

//template <>
/*float Sinogram3D_GPU_half::getSinogramL1Norm()
{
	half* currentSinogramData = this->getDataSinogram();
	float* g_odata;
	float l1Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(float),cudaMemAttachGlobal));
	sinogramReduce_half<half,float,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

//template <>
float Sinogram3D_GPU_half::getSinogramL2Norm()
{
	half* currentSinogramData = this->getDataSinogram();
	float* g_odata;
	float l2Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(float),cudaMemAttachGlobal));
	sinogramReduce_square_half<half,float,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += g_odata[i];

	return l2Norm;
}

//template <>
float Sinogram3D_GPU_half::getSinogramLpNorm(float power)
{
	half* currentSinogramData = this->getDataSinogram();
	float* g_odata;
	float lpNorm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(float),cudaMemAttachGlobal));
	sinogramReduce_abspow_half<half,float,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];

	return lpNorm;
}

//template <>
float Sinogram3D_GPU_half::getSinogramMean()
{
	float sinogramMean = this->getSinogramL1Norm<float>();
	sinogramMean/=(float)this->getDataSinogramSize();

	return sinogramMean;
}

//template <>
float Sinogram3D_GPU_half::getSinogramMeanSquare()
{
	float sinogramMeanSquare = this->getSinogramL2Norm<float>();
	sinogramMeanSquare/=(float)this->getDataSinogramSize();

	return sinogramMeanSquare;
}

//template <>
float Sinogram3D_GPU_half::getSinogramStd()
{
	float sinogramMean = this->getSinogramMean<float>();
	half* currentSinogramData = this->getDataSinogram();
	float* g_odata;
	float sinogramStd=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(float),cudaMemAttachGlobal));
	sinogramReduce_std_half<half,float,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, sinogramMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		sinogramStd += g_odata[i];

	sinogramStd/=(float)this->getDataSinogramSize();

	return sqrt(sinogramStd);
}
 */
//template <>
double Sinogram3D_GPU_half::getSinogramL1Norm()
{
	half* currentSinogramData = this->getDataSinogram();
	double* g_odata;
	double l1Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(double),cudaMemAttachGlobal));
	sinogramReduce_half<half,double,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

//template <>
double Sinogram3D_GPU_half::getSinogramL2Norm()
{
	half* currentSinogramData = this->getDataSinogram();
	double* g_odata;
	double l2Norm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(double),cudaMemAttachGlobal));
	sinogramReduce_square_half<half,double,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += g_odata[i];

	return l2Norm;
}

//template <>
double Sinogram3D_GPU_half::getSinogramLpNorm(double power)
{
	half* currentSinogramData = this->getDataSinogram();
	double* g_odata;
	double lpNorm=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(double),cudaMemAttachGlobal));
	sinogramReduce_abspow_half<half,double,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];

	return lpNorm;
}

//template <>
double Sinogram3D_GPU_half::getSinogramMean()
{
	double sinogramMean = this->getSinogramL1Norm();
	sinogramMean/=(double)this->getDataSinogramSize();

	return sinogramMean;
}

//template <>
double Sinogram3D_GPU_half::getSinogramMeanSquare()
{
	double sinogramMeanSquare = this->getSinogramL2Norm();
	sinogramMeanSquare/=(double)this->getDataSinogramSize();

	return sinogramMeanSquare;
}

//template <>
double Sinogram3D_GPU_half::getSinogramStd()
{
	double sinogramMean = this->getSinogramMean();
	half* currentSinogramData = this->getDataSinogram();
	double* g_odata;
	double sinogramStd=0;
	unsigned long int n = this->getDataSinogramSize();
	int maxThreads = 1024;  // number of threads per block
	int maxBlocks = 64;

	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	if((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
		cout << "n is too large, please choose a smaller number!" << endl;

	if(blocks > prop.maxGridSize[0])
	{
		cout << "Grid size <" << blocks << "> exceeds the device capability <" << prop.maxGridSize[0] << ">, set block size as " << threads*2 << "(original" << threads << "%d)" << endl;
		blocks /= 2;
		threads *= 2;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

	checkCudaErrors(cudaMallocManaged ((void **)&g_odata, blocks*sizeof(double),cudaMemAttachGlobal));
	sinogramReduce_std_half<half,double,512,true><<<dimGrid, dimBlock, smemSize>>>(currentSinogramData, g_odata, n, sinogramMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		sinogramStd += g_odata[i];

	sinogramStd/=(double)this->getDataSinogramSize();

	return sqrt(sinogramStd);
}

#include "Sinogram3D_instances.cu"
#include "Sinogram3D_instances_GPU.cu"