#include "Image3D_GPU.cuh"
#include "GPUConstant.cuh"
#include "Image3D_GPU_kernel_half.cuh"
#include "Image3D_GPU_kernel.cuh"
#include "half_float_conversion_kernel.cuh"
#include "Acquisition.hpp"
//#include "Volume.cuh"

template <typename T>
Image3D_GPU<T>::Image3D_GPU() : Image3D<T>(){}

template <typename T>
Image3D_GPU<T>::Image3D_GPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture) : Image3D<T>(xImagePixelNb, yImagePixelNb, zImagePixelNb)
{
	T* dataImageTmp ;
	cout << "Allocation dans Image3D_GPU : " <<this->getDataImageSize() << endl;
	this->setCUDAArchitecture(cudaArchitecture);
	checkCudaErrors(cudaMallocManaged ((void **)&dataImageTmp, sizeof(T)*this->getDataImageSize(),cudaMemAttachGlobal));
	checkCudaErrors(cudaMemset(dataImageTmp,0.0,sizeof(T)*this->getDataImageSize()));
	this->setImageData(dataImageTmp);
	this->copyConstantGPU();
}

template <>
Image3D_GPU<half>::Image3D_GPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture) : Image3D<half>(xImagePixelNb, yImagePixelNb, zImagePixelNb)
{
	half* dataImageTmp ;
	this->setCUDAArchitecture(cudaArchitecture);
	checkCudaErrors(cudaMallocManaged ((void **)&dataImageTmp, sizeof(half)*this->getDataImageSize(),cudaMemAttachGlobal));
	checkCudaErrors(cudaMemset(dataImageTmp,0.0,sizeof(half)*this->getDataImageSize()));
	this->setImageData(dataImageTmp);
	this->copyConstantGPU();
}

template <typename T>
Image3D_GPU<T>::~Image3D_GPU()
{
	T* dataImageTmp = this->getImageData();
	checkCudaErrors(cudaFree ((void *)dataImageTmp));
}

template <>
Image3D_GPU<half>::~Image3D_GPU()
{
	half* dataImageTmp = this->getImageData();
	checkCudaErrors(cudaFree ((void *)dataImageTmp));
}


template <typename T>
CUDAArchitecture* Image3D_GPU<T>::getCUDAArchitecture() const
{
	return this->cudaArchitecture;
}

template <typename T>
void Image3D_GPU<T>::setCUDAArchitecture(CUDAArchitecture* cudaArchitecture)
{
	this->cudaArchitecture=cudaArchitecture;
}

template <typename T>
Image3D_GPU<T>::Image3D_GPU(const Image3D_GPU<T>& imageToCopy)
{
	unsigned long int x,y,z,xNb,yNb,zNb;

	this->setXImagePixelNb(imageToCopy.getXImagePixelNb());
	this->setYImagePixelNb(imageToCopy.getYImagePixelNb());
	this->setZImagePixelNb(imageToCopy.getZImagePixelNb());

	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	//this->cudaArchitecture = new CUDAArchitecture();
	this->setCUDAArchitecture(imageToCopy.getCUDAArchitecture());

	this->setDataImageSize(xNb,yNb,zNb);
	T* dataImageTmp = this->getImageData();
	checkCudaErrors(cudaMallocManaged ((void **)&dataImageTmp, sizeof(T)*this->getDataImageSize(),cudaMemAttachGlobal));
	cudaMemset(dataImageTmp,0.0,sizeof(T)*this->getDataImageSize());
	this->setImageData(dataImageTmp);

	T* currentImageData = this->getImageData();
	T* imageToCopyData = imageToCopy.getImageData();

	for (z=0;z<zNb;z++)
		for (y=0;y<yNb;y++)
			for (x=0;x<xNb;x++)
				currentImageData[x+y*xNb+z*xNb*yNb]=imageToCopyData[x+y*xNb+z*xNb*yNb];
}

template <typename T>
void Image3D_GPU<T>::copyImage3D(Image3D_GPU<T>* imageToCopy)
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

template <typename T>
void Image3D_GPU<T>::setImage(T value)
{
	T* currentImageData = this->getImageData();

	dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
	dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

	setImage_k<T><<<dimGrid,dimBlock>>>(currentImageData, value, this->getDataImageSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
void Image3D_GPU<T>::scalarImage(T value)
{
	T* currentImageData = this->getImageData();

	dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
	dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

	scalarImage_k<T><<<dimGrid,dimBlock>>>(currentImageData, value, this->getDataImageSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
void Image3D_GPU<T>::addImage(Image3D_GPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		addImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::addImage(Image3D_GPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		addImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize(), lambda);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::positiveAddImage(Image3D_GPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		positiveAddImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize(), lambda);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::diffImage(Image3D_GPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		diffImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::diffImage(T lambda, Image3D_GPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		diffImage_k<T><<<dimGrid,dimBlock>>>(lambda, currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::diffImage(Image3D_GPU<T>* image2, T lambda)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		diffImage_k_moins_lambda<T><<<dimGrid,dimBlock>>>(lambda, currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::multImage(Image3D_GPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		multImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
T Image3D_GPU<T>::scalarProductImage(Image3D_GPU<T>* image2)
{
	if(this->isSameSize(image2))
	{
		T* currentImageData = this->getImageData();
		T* image2Data = image2->getImageData();
		T* g_odata;
		T scalarProduct=0;
		unsigned long int n = this->getDataImageSize();
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
		imageReduce_scalarProduct<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, image2Data, g_odata, n);
		checkCudaErrors(cudaDeviceSynchronize());

		// Final reducing
		for(unsigned int i = 0; i<blocks; i++)
			scalarProduct += g_odata[i];

		return scalarProduct;

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
T Image3D_GPU<T>::getImageL1Norm()
{
	T* currentImageData = this->getImageData();
	T* g_odata;
	T l1Norm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

template <typename T>
T Image3D_GPU<T>::getImageL2Norm()
{
	T* currentImageData = this->getImageData();
	T* g_odata;
	double l2Norm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_square<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += (double)g_odata[i];

	return (float)l2Norm;
}

template <typename T>
T Image3D_GPU<T>::getImageLpNorm(T power)
{
	T* currentImageData = this->getImageData();
	T* g_odata;
	T lpNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_abspow<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];

	return lpNorm;
}

template <typename T>
T Image3D_GPU<T>::getImageHuberNorm(T threshold)
{
	T* currentImageData = this->getImageData();
	T* g_odata;
	T huberNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_huber<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, threshold);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		huberNorm += g_odata[i];

	return huberNorm;
}

template <typename T>
T Image3D_GPU<T>::getImageMean()
{
	T imageMean = this->getImageL1Norm();
	imageMean/=(T)this->getDataImageSize();

	return imageMean;
}

template <typename T>
T Image3D_GPU<T>::getImageMeanSquare()
{
	T imageMeanSquare = this->getImageL2Norm();
	imageMeanSquare/=(T)this->getDataImageSize();

	return imageMeanSquare;
}

template <typename T>
T Image3D_GPU<T>::getImageStd()
{
	T imageMean = this->getImageMean();
	T* currentImageData = this->getImageData();
	T* g_odata;
	T imageStd=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_std<T,256,false><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, imageMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		imageStd += g_odata[i];


	imageStd/=(T)this->getDataImageSize();

	return sqrt(imageStd);
}

template <typename T>
void Image3D_GPU<T>::getImageSign(Image3D_GPU<T>* signedImage)
{
	if(this->isSameSize(signedImage))
	{
		T* currentImageData = this->getImageData();
		T* signedImageData = signedImage->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		signedImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,signedImageData, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void Image3D_GPU<T>::getImageAbsPow(Image3D_GPU<T>* absPowImage, T p)
{
	if(this->isSameSize(absPowImage))
	{
		T* currentImageData = this->getImageData();
		T* absPowImageData = absPowImage->getImageData();

		dim3 dimBlock(this->getCUDAArchitecture()->getXThreadNb(), this->getCUDAArchitecture()->getYThreadNb(), this->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(this->getCUDAArchitecture()->getXBlockNb(), this->getCUDAArchitecture()->getYBlockNb(), this->getCUDAArchitecture()->getZBlockNb());

		absPowImage_k<T><<<dimGrid,dimBlock>>>(currentImageData,absPowImageData, this->getDataImageSize(), p);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}


/* Copy Sinogram_GPU constant */
template <typename T>
__host__ void Image3D_GPU<T>::copyConstantGPU()
{
	unsigned long int xImagePixelNb = this->getXImagePixelNb();
	unsigned long int yImagePixelNb = this->getYImagePixelNb();

	cudaMemcpyToSymbol(xVolumePixelNb_GPU,&xImagePixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(yVolumePixelNb_GPU,&yImagePixelNb,sizeof(unsigned long int));
}


/* Image3D_GPU_half definition */
Image3D_GPU_half::Image3D_GPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitecture) : Image3D_GPU<half>(xImagePixelNb, yImagePixelNb, zImagePixelNb,cudaArchitecture){}
Image3D_GPU_half::~Image3D_GPU_half(){}

void Image3D_GPU_half::copyImage3D( Image3D_GPU_half* imageToCopy)
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

Image3D_GPU_half::Image3D_GPU_half(const Image3D_GPU_half& imageToCopy)
{
	this->setXImagePixelNb(imageToCopy.getXImagePixelNb());
	this->setYImagePixelNb(imageToCopy.getYImagePixelNb());
	this->setZImagePixelNb(imageToCopy.getZImagePixelNb());


	unsigned long int x,y,z,xNb,yNb,zNb;
	xNb = this->getXImagePixelNb();
	yNb = this->getYImagePixelNb();
	zNb = this->getZImagePixelNb();

	this->setDataImageSize(xNb,yNb,zNb);
	half* dataImageTmp = this->getImageData();
	checkCudaErrors(cudaMallocManaged ((void **)&dataImageTmp, sizeof(half)*this->getDataImageSize(),cudaMemAttachGlobal));
	cudaMemset(dataImageTmp,0.0,sizeof(half)*this->getDataImageSize());
	this->setImageData(dataImageTmp);

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

Image3D_GPU_half & Image3D_GPU_half::operator=(const Image3D_GPU_half &image)
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

void Image3D_GPU_half::saveImage(string fileName)
{
	ofstream imageFile;
	imageFile.open(fileName.c_str(),ios::out | ios::binary);

	if (imageFile.is_open())
	{
		int i;
		cout << "Saving " << fileName << " image" << endl;
		half* currentImageData = this->getImageData();

		Image3D_GPU<float>* vol_32f = new Image3D_GPU<float>(this->getXImagePixelNb(), this->getYImagePixelNb(), this->getZImagePixelNb(),this->getCUDAArchitecture());


		//this->setCUDAArchitecture(new CUDAArchitecture(1, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short zBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short zThreadNb));

		dim3 dimBlock(vol_32f->getCUDAArchitecture()->getXThreadNb(), vol_32f->getCUDAArchitecture()->getYThreadNb(), vol_32f->getCUDAArchitecture()->getZThreadNb());
		dim3 dimGrid(vol_32f->getCUDAArchitecture()->getXBlockNb(), vol_32f->getCUDAArchitecture()->getYBlockNb(), vol_32f->getCUDAArchitecture()->getZBlockNb());
		cout << "beginnning compute "  << endl;
		half_to_float_kernel<<<dimGrid,dimBlock>>>(currentImageData,vol_32f->getImageData());
		checkCudaErrors(cudaDeviceSynchronize());
		cout << "end compute "  << endl;

		imageFile.write((char*)vol_32f->getImageData(),sizeof(float)*this->getDataImageSize());
		imageFile.close();
		cout << "end write "  << endl;
		//delete vol_32f;
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

void Image3D_GPU_half::saveMiddleSliceImage(string fileName)
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

void Image3D_GPU_half::loadImage(string fileName)
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

void Image3D_GPU_half::setImage(float value)
{
	half* currentImageData = this->getImageData();

	CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

	dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
	dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

	setImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData, value, this->getDataImageSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

void Image3D_GPU_half::scalarImage(float value)
{
	half* currentImageData = this->getImageData();

	CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

	dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
	dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

	scalarImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData, value, this->getDataImageSize());
	checkCudaErrors(cudaDeviceSynchronize());
}

void Image3D_GPU_half::addImage(Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		addImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_GPU_half::addImage(Image3D_GPU_half* image2, float lambda)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		addImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize(), lambda);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_GPU_half::positiveAddImage(Image3D_GPU_half* image2, float lambda)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		addImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize(), lambda);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_GPU_half::diffImage(Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		diffImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_GPU_half::diffImage(float lambda, Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		diffImage_k_half<half><<<dimGrid,dimBlock>>>(lambda, currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void Image3D_GPU_half::multImage(Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		multImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,image2Data, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template<>
float Image3D_GPU_half::scalarProductImage(Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		float* g_odata;
		float scalarProduct=0;
		unsigned long int n = this->getDataImageSize();
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
		imageReduce_scalarProduct_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, image2Data, g_odata, n);
		checkCudaErrors(cudaDeviceSynchronize());

		// Final reducing
		for(unsigned int i = 0; i<blocks; i++)
			scalarProduct += g_odata[i];

		return scalarProduct;

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template<>
float Image3D_GPU_half::getImageL1Norm()
{
	half* currentImageData = this->getImageData();
	float* g_odata;
	float l1Norm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

template<>
float Image3D_GPU_half::getImageL2Norm()
{
	half* currentImageData = this->getImageData();
	float* g_odata;
	float l2Norm=0.0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_square_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += g_odata[i];

	return l2Norm;
}

template<>
float Image3D_GPU_half::getImageLpNorm(float power)
{
	half* currentImageData = this->getImageData();
	float* g_odata;
	float lpNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_abspow_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];


	return lpNorm;
}

template <>
float Image3D_GPU_half::getImageHuberNorm(float threshold)
{
	half* currentImageData = this->getImageData();
	float* g_odata;
	float huberNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_huber_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, threshold);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		huberNorm += g_odata[i];

	return huberNorm;
}

template<>
float Image3D_GPU_half::getImageMean()
{
	float imageMean = this->getImageL1Norm<float>();
	imageMean/=(float)this->getDataImageSize();

	return imageMean;
}

template<>
float Image3D_GPU_half::getImageMeanSquare()
{
	float imageMeanSquare = this->getImageL2Norm<float>();
	imageMeanSquare/=(float)this->getDataImageSize();

	return imageMeanSquare;
}

template<>
float Image3D_GPU_half::getImageStd()
{
	float imageMean = this->getImageMean<float>();
	half* currentImageData = this->getImageData();
	float* g_odata;
	float imageStd=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_std_half<half,float,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, imageMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		imageStd += g_odata[i];

	imageStd/=(float)this->getDataImageSize();

	return sqrt(imageStd);
}

template<>
double Image3D_GPU_half::scalarProductImage(Image3D_GPU_half* image2)
{
	if(this->isSameSize(image2))
	{
		half* currentImageData = this->getImageData();
		half* image2Data = image2->getImageData();
		double* g_odata;
		double scalarProduct=0;
		unsigned long int n = this->getDataImageSize();
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
		imageReduce_scalarProduct_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, image2Data, g_odata, n);
		checkCudaErrors(cudaDeviceSynchronize());

		// Final reducing
		for(unsigned int i = 0; i<blocks; i++)
			scalarProduct += g_odata[i];

		return scalarProduct;

	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template<>
double Image3D_GPU_half::getImageL1Norm()
{
	half* currentImageData = this->getImageData();
	double* g_odata;
	double l1Norm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l1Norm += g_odata[i];

	return l1Norm;
}

template<>
double Image3D_GPU_half::getImageL2Norm()
{
	half* currentImageData = this->getImageData();
	double* g_odata;
	double l2Norm=0.0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_square_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		l2Norm += g_odata[i];

	return l2Norm;
}

template<>
double Image3D_GPU_half::getImageLpNorm(double power)
{
	half* currentImageData = this->getImageData();
	double* g_odata;
	double lpNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_abspow_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, power);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		lpNorm += g_odata[i];


	return lpNorm;
}

template <>
double Image3D_GPU_half::getImageHuberNorm(double threshold)
{
	half* currentImageData = this->getImageData();
	double* g_odata;
	double huberNorm=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_huber_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, threshold);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		huberNorm += g_odata[i];

	return huberNorm;
}

template<>
double Image3D_GPU_half::getImageMean()
{
	double imageMean = this->getImageL1Norm<double>();
	imageMean/=(double)this->getDataImageSize();

	return imageMean;
}

template<>
double Image3D_GPU_half::getImageMeanSquare()
{
	double imageMeanSquare = this->getImageL2Norm<double>();
	imageMeanSquare/=(double)this->getDataImageSize();

	return imageMeanSquare;
}

template<>
double Image3D_GPU_half::getImageStd()
{
	double imageMean = this->getImageMean<double>();
	half* currentImageData = this->getImageData();
	double* g_odata;
	double imageStd=0;
	unsigned long int n = this->getDataImageSize();
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
	imageReduce_std_half<half,double,256,true><<<dimGrid, dimBlock, smemSize>>>(currentImageData, g_odata, n, imageMean);
	checkCudaErrors(cudaDeviceSynchronize());

	// Final reducing
	for(unsigned int i = 0; i<blocks; i++)
		imageStd += g_odata[i];

	imageStd/=(double)this->getDataImageSize();

	return sqrt(imageStd);
}

void Image3D_GPU_half::getImageSign(Image3D_GPU_half* signedImage)
{
	if(this->isSameSize(signedImage))
	{
		half* currentImageData = this->getImageData();
		half* signedImageData = signedImage->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		signedImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,signedImageData, this->getDataImageSize());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <>
void Image3D_GPU_half::getImageAbsPow(Image3D_GPU_half* absPowImage, float p)
{
	if(this->isSameSize(absPowImage))
	{
		half* currentImageData = this->getImageData();
		half* absPowImageData = absPowImage->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		absPowImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,absPowImageData, this->getDataImageSize(), p);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <>
void Image3D_GPU_half::getImageAbsPow(Image3D_GPU_half* absPowImage, double p)
{
	if(this->isSameSize(absPowImage))
	{
		half* currentImageData = this->getImageData();
		half* absPowImageData = absPowImage->getImageData();

		CUDAArchitecture* gpuArch = (((Image3D_GPU<float>*)this)->getCUDAArchitecture());

		dim3 dimBlock(gpuArch->getXThreadNb(), gpuArch->getYThreadNb(), gpuArch->getZThreadNb());
		dim3 dimGrid(gpuArch->getXBlockNb(), gpuArch->getYBlockNb(), gpuArch->getZBlockNb());

		absPowImage_k_half<half><<<dimGrid,dimBlock>>>(currentImageData,absPowImageData, this->getDataImageSize(), p);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		cout << "Images must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}


CUDAArchitecture* Image3D_GPU_half::getCUDAArchitecture() const
{
	return this->cudaArchitecture;
}


void Image3D_GPU_half::setCUDAArchitecture(CUDAArchitecture* cudaArchitecture)
{
	this->cudaArchitecture=cudaArchitecture;
}

#include "Image3D_instances.cu"
#include "Image3D_instances_GPU.cu"
