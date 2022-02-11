/*
 * 3DIterativeReconstruction.cu
 *
 */

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <omp.h>
#include "ConfigTiff.hpp"
#include "ConfigCT.hpp"
#include "ConfigComputeArchitecture.hpp"
#include "ConfigIteration.hpp"
#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"
#include "Image3D.cuh"
#include "Image3D_CPU.cuh"
#include "Image3D_GPU.cuh"
//#include "Image3D_MGPU.cuh"
#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Projector.cuh"
#include "Projector_CPU.cuh"
#include "Projector_GPU.cuh"
//#include "Projector_MGPU.cuh"
#include "BackProjector.cuh"
#include "BackProjector_CPU.cuh"
#include "BackProjector_GPU.cuh"
//#include "BackProjector_MGPU.cuh"
#include "ComputingArchitecture.cuh"
#include "Convolution3D.cuh"
#include "Convolution3D_CPU.cuh"
#include "Convolution3D_GPU.cuh"
//#include "Convolution3D_MGPU.cuh"
#include "HuberRegularizer_CPU.cuh"
#include "HuberRegularizer_GPU.cuh"
//#include "HuberRegularizer_MGPU.cuh"
#include "GeneralizedGaussianRegularizer_CPU.cuh"
#include "GeneralizedGaussianRegularizer_GPU.cuh"
//#include "GeneralizedGaussianRegularizer_MGPU.cuh"
#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
//#include "Iter3D_MGPU.cuh"


std::mutex iomutex;
std::atomic<int> counter;

#ifdef __CUDACC__

texture<float,cudaTextureType2DLayered> sinogram_tex0;
texture<float, cudaTextureType3D, cudaReadModeElementType> volume_tex;


/* Regular Sampling projector constant */


__device__ __constant__ float alphaIOcylinderC_GPU[MAX_PROJECTION];
__device__ __constant__ float betaIOcylinderC_GPU[MAX_PROJECTION];
__device__ __constant__ float gammaIOcylinderC_GPU;
__device__ __constant__ float xVolumeCenterPixel_GPU;
__device__ __constant__ float yVolumeCenterPixel_GPU;
__device__ __constant__ float zVolumeCenterPixel_GPU;
__device__ __constant__ float xVolumePixelSize_GPU;
__device__ __constant__ unsigned long int xVolumePixelNb_GPU;
__device__ __constant__ unsigned long int yVolumePixelNb_GPU;
__device__ __constant__ unsigned long int zVolumePixelNb_GPU;




/* Joseph projector constant */
#ifdef JOSEPH
__device__ __constant__ float alphaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float betaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float deltaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float sigmaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float kappaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float iotaPreComputingC_GPU[MAX_PROJECTION];
__device__ __constant__ float gammaPrecomputingC_GPU;
__device__ __constant__ float omegaPrecomputingC_GPU;
#endif

/* FDK BackProjection constant*/

__device__ __constant__ float alphaC_GPU;
__device__ __constant__ float betaC_GPU;




/* Convolution constant */
__device__ __constant__  float c_Kernel_h[MAX_SIZE_H_KERNEL];
__device__ __constant__  float c_Kernel_v[MAX_SIZE_V_KERNEL];
__device__ __constant__  float c_Kernel_p[MAX_SIZE_P_KERNEL];
__device__ __constant__  unsigned long int c_volume_x;
__device__ __constant__  unsigned long int c_volume_y;
__device__ __constant__  unsigned long int c_volume_z;
__device__ __constant__  unsigned long int c_kernel_radius_x;
__device__ __constant__  unsigned long int c_kernel_radius_y;
__device__ __constant__  unsigned long int c_kernel_radius_z;



/* Acquisition constant */

__device__ __constant__ float focusDetectorDistance_GPU;
__device__ __constant__ float focusObjectDistance_GPU;


/* Detector constant */

__device__ __constant__ float uDetectorCenterPixel_GPU;
__device__ __constant__ float vDetectorCenterPixel_GPU;
__device__ __constant__ float uDetectorPixelSize_GPU;
__device__ __constant__ float vDetectorPixelSize_GPU;
__device__ __constant__ unsigned long int uDetectorPixelNb_GPU;
__device__ __constant__ unsigned long int vDetectorPixelNb_GPU;
__device__ __constant__ unsigned long int projectionNb_GPU;


/* Sinogram_GPU constant */


__device__ __constant__ unsigned long int uSinogramPixelNb_GPU;
__device__ __constant__ unsigned long int vSinogramPixelNb_GPU;


/* Segmentation GPU constant */
//Nicolas

__device__ __constant__ int Kclasse;
__device__ __constant__ double gammaPotts;//Potts coefficient
__device__ __constant__ double meanclasses[MAX_CLASSE];//means of the classes
__device__ __constant__ double varianceclasses[MAX_CLASSE];//variances of the classes
__device__ __constant__ double energySingleton[MAX_CLASSE];//energies of singleton


#endif


int main( int argc, char** argv)
{
#ifdef __linux__
	std::cout << "\tMain running on CPU " << sched_getcpu() <<  std::endl;
#endif

	 /*unsigned num_cpus = std::thread::hardware_concurrency();
	      std::cout << "Launching " << num_cpus << " threads\n";

	      // A mutex ensures orderly access to std::cout from multiple threads.
	      std::mutex iomutex;
	      std::vector<std::thread> threads(num_cpus);
	      for (unsigned i = 0; i < num_cpus; ++i) {
	        threads[i] = std::thread([&iomutex, i] {
	          {
	            // Use a lexical scope and lock_guard to safely lock the mutex only for
	            // the duration of std::cout usage.
	            std::lock_guard<std::mutex> iolock(iomutex);
	            std::cout << "Thread #" << i << " is running\n";
	          }

	          // Simulate important work done by the tread by sleeping for a bit...
	          std::this_thread::sleep_for(std::chrono::milliseconds(200));

	        });
	      }

	      for (auto& t : threads) {
	        t.join();
	      }*/


	if (argc<2)
	{printf("En arguments le nombre de projections !\n");
		return -1;
	}
	char workdirectory[180];
	sprintf(workdirectory,"Projections_%u/",(unsigned int)std::strtoul(argv[1],NULL,10));
	cout << workdirectory << endl;

	ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"));
	//configComputeArchitectureFile=new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"));
	char *precision_string = NULL;
	char *storage_string = NULL;
	char *compute_string = NULL;
	char *pair_string = NULL;
	char *architecture_string = NULL;
	int ngpu,nbstreams;
	if (getCmdLineArgumentString(argc, (const char**)argv, "precision", &precision_string)){
		if (strcmp(precision_string,"half")==0)
		configComputeArchitectureFile->setPrecision(HALF_GPU);
	else if (strcmp(precision_string,"double")==0)
	configComputeArchitectureFile->setPrecision(DOUBLE_GPU);
	else
	configComputeArchitectureFile->setPrecision(FLOAT_GPU);
		}

		if (getCmdLineArgumentString(argc, (const char**)argv, "storage", &storage_string)){
			if (strcmp(storage_string,"mem_GPU")==0)
			configComputeArchitectureFile->setStorage(MEM_GPU);
	else if (strcmp(storage_string,"mem_MGPU")==0)
	configComputeArchitectureFile->setStorage(MEM_MGPU);
	else
	configComputeArchitectureFile->setStorage(MEM_CPU);
			
		}


		if (getCmdLineArgumentString(argc, (const char**)argv, "architecture", &architecture_string)){
			if (strcmp(architecture_string,"CPU")==0)
			configComputeArchitectureFile->setArchitecture(ARCHITECTURE_CPU);
	else if (strcmp(architecture_string,"FPGA")==0)
	configComputeArchitectureFile->setArchitecture(ARCHITECTURE_FPGA);
	else
	configComputeArchitectureFile->setArchitecture(ARCHITECTURE_GPU);
		}

		if (getCmdLineArgumentString(argc, (const char**)argv, "pair", &pair_string)){
			if (strcmp(pair_string,"SiddonVI")==0)
			configComputeArchitectureFile->setPair(PAIR_SIDDONVI);
	else if (strcmp(pair_string,"SFTR")==0){
	configComputeArchitectureFile->setPair(PAIR_SFTR);}
	else
	configComputeArchitectureFile->setPair(PAIR_RSVI);
		}



		if (getCmdLineArgumentString(argc, (const char**)argv, "compute", &compute_string)){
			if (strcmp(compute_string,"OCL")==0)
			configComputeArchitectureFile->setCompute(COMPUTE_OCL);
	else if (strcmp(compute_string,"CPU")==0)
	configComputeArchitectureFile->setCompute(COMPUTE_C);
	else
	configComputeArchitectureFile->setCompute(COMPUTE_CUDA);
		}

	if (nbstreams=getCmdLineArgumentInt(argc, (const char**)argv, "nstreams")){
		configComputeArchitectureFile->setProjectionStreamsNb(nbstreams);
		configComputeArchitectureFile->setBProjectionStreamsNb(nbstreams);
		}

	else{
	configComputeArchitectureFile->setProjectionStreamsNb(configComputeArchitectureFile->getConfigFileField<int>("streamsNbP"));
	configComputeArchitectureFile->setBProjectionStreamsNb(configComputeArchitectureFile->getConfigFileField<int>("streamsNbBP"));
	}

	if(ngpu=getCmdLineArgumentInt(argc, (const char**)argv, "ngpu")){
		configComputeArchitectureFile->setGpuNb_proj(ngpu);
		configComputeArchitectureFile->setGpuNb_back(ngpu);
		configComputeArchitectureFile->setGpuNb_sino(ngpu);
		configComputeArchitectureFile->setGpuNb_vol(ngpu);

	}
		else{
		configComputeArchitectureFile->setGpuNb_proj(configComputeArchitectureFile->getConfigFileField<int>("gpuNbP"));
		configComputeArchitectureFile->setGpuNb_back(configComputeArchitectureFile->getConfigFileField<int>("gpuNbBP"));
		configComputeArchitectureFile->setGpuNb_sino(configComputeArchitectureFile->getConfigFileField<int>("gpuNbSino"));
		configComputeArchitectureFile->setGpuNb_vol(configComputeArchitectureFile->getConfigFileField<int>("gpuNbVolume"));
		}

		printf("--pair %d\n",configComputeArchitectureFile->getPair());
		printf("--compute %d\n",configComputeArchitectureFile->getCompute());

	switch(configComputeArchitectureFile->getPrecision()){
	case HALF_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :
		{
			Iter3D_CPU_half* iter = new Iter3D_CPU_half(workdirectory,configComputeArchitectureFile);
			iter->doMainIter(argc,argv);
		}
		break;

		case MEM_GPU :
		{
			Iter3D_GPU_half* iter = new Iter3D_GPU_half(workdirectory,configComputeArchitectureFile);
			iter->doMainIter(argc,argv);
		}
		break;

		case MEM_MGPU :{

		}
		break;


		}

		break;
	}

	case FLOAT_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			switch(configComputeArchitectureFile->getPair()){
				case PAIR_RSVI : {
					switch(configComputeArchitectureFile->getCompute()){
						case COMPUTE_C : {Iter3D_RSVI_compute_C_mem_CPU<float>* iter = new Iter3D_RSVI_compute_C_mem_CPU<float>(workdirectory,configComputeArchitectureFile);
						iter->doMainIter(argc,argv);}
						break;
						case COMPUTE_CUDA : {Iter3D_RSVI_compute_CUDA_mem_CPU<float>* iter = new Iter3D_RSVI_compute_CUDA_mem_CPU<float>(workdirectory,configComputeArchitectureFile);
						iter->doMainIter(argc,argv);}
						break;
						case COMPUTE_OCL : {Iter3D_RSVI_compute_OCL_mem_CPU<float>* iter = new Iter3D_RSVI_compute_OCL_mem_CPU<float>(workdirectory,configComputeArchitectureFile);
						iter->doMainIter(argc,argv);}
						break;}
						break;
					}
				case PAIR_SIDDONVI : {
					switch(configComputeArchitectureFile->getCompute()){
						case COMPUTE_C : {Iter3D_SiddonVI_compute_C_mem_CPU<float>* iter = new Iter3D_SiddonVI_compute_C_mem_CPU<float>(workdirectory,configComputeArchitectureFile);
						iter->doMainIter(argc,argv);}
						break;
						case COMPUTE_OCL : {Iter3D_SiddonVI_compute_OCL_mem_CPU<float>* iter = new Iter3D_SiddonVI_compute_OCL_mem_CPU<float>(workdirectory,configComputeArchitectureFile);
						iter->doMainIter(argc,argv);}
						break;
					}
					break;
				}
			}
			
			
		}
		break;

		case MEM_GPU :{
			Iter3D_RSVI_compute_CUDA_mem_GPU<float>* iter = new Iter3D_RSVI_compute_CUDA_mem_GPU<float>(workdirectory,configComputeArchitectureFile);
			iter->doMainIter(argc,argv);
		}
		break;

		case MEM_MGPU :{
			//Iter3D_RSVI_compute_CUDA_mem_MGPU<float>* iter = new Iter3D_RSVI_compute_CUDA_mem_MGPU<float>(workdirectory,configComputeArchitectureFile);
			//iter->doMainIter(argc,argv);
		}
		break;
		}

		break;}

	case DOUBLE_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			Iter3D_RSVI_compute_CUDA_mem_CPU<double>* iter = new Iter3D_RSVI_compute_CUDA_mem_CPU<double>(workdirectory,configComputeArchitectureFile);
			iter->doMainIter(argc,argv);
		}
		break;
		case MEM_GPU :{
			Iter3D_RSVI_compute_CUDA_mem_GPU<double>* iter = new Iter3D_RSVI_compute_CUDA_mem_GPU<double>(workdirectory,configComputeArchitectureFile);
			iter->doMainIter(argc,argv);
		}
		break;

		case MEM_MGPU :{

		}
		break;
		}
		break;}
	}






	return 0;
}


