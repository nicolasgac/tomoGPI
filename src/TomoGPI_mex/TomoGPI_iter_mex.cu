#include "mex.h"
#include "class_handle_iter.hpp"

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include "ConfigTiff.hpp"
#include "ConfigCT.hpp"
#include "ConfigComputeArchitecture.hpp"
#include "ConfigIteration.hpp"
#include "Sinogram3D.cuh"
#include "Image3D.cuh"
#include "Volume.cuh"
#include "Projector.cuh"
#include "BackProjector.cuh"
#include "ComputingArchitecture.cuh"
#include "Convolution3D.cuh"
#include "HuberRegularizer_CPU.cuh"
#include "HuberRegularizer_GPU.cuh"
#include "GeneralizedGaussianRegularizer_CPU.cuh"
#include "GeneralizedGaussianRegularizer_GPU.cuh"
#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
#include "Mex.cuh"
#include <cuda_fp16.h>

// The class that we are interfacing to
/*class dummy
{
public:
    void train() {};
    void test() {};
private:
};*/


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



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	char cmd[128];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

	if (!strcmp("new", cmd)) {
		// Check parameters
		if (nlhs != 1)
			mexErrMsgTxt("New: One output expected.");

		char workdirectory[180];
		/*	sprintf(workdirectory,"Projections_%u/",(unsigned int)std::strtoul(mxArrayToString(prhs[1]),NULL,10));
			cout << workdirectory << endl;*/

		mxGetString(prhs[1], workdirectory, sizeof(workdirectory));

		ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture(string(workdirectory), string("configComputeArchitecture"));


		switch(configComputeArchitectureFile->getPrecision()){
		case HALF_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :
			{
				Mex_CPU_half* mex = new Mex_CPU_half(std::string(workdirectory));
				plhs[0] = convertPtr2Mat<Mex_CPU_half >(mex);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;

			case MEM_GPU :
			{
				Mex_GPU_half* mex = new Mex_GPU_half(std::string(workdirectory));
				plhs[0] = convertPtr2Mat<Mex_GPU_half >(mex);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}

			break;
		}

		case FLOAT_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				printf("coucou 3\n");
				Mex_CPU<float>* mex = new Mex_CPU<float>(std::string(workdirectory));
				printf("coucou 4\n");
				plhs[0] = convertPtr2Mat<Mex_CPU<float> >(mex);
				printf("coucou 5\n");
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;

			case MEM_GPU :{
				Mex_GPU<float>* mex = new Mex_GPU<float>(std::string(workdirectory));
				plhs[0] = convertPtr2Mat<Mex_GPU<float> >(mex);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}

			break;}

		case DOUBLE_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				Mex_CPU<double>* mex = new Mex_CPU<double>(std::string(workdirectory));
				plhs[0] = convertPtr2Mat<Mex_CPU<double> >(mex);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			case MEM_GPU :{
				Mex_GPU<double>* mex = new Mex_GPU<double>(std::string(workdirectory));
				plhs[0] = convertPtr2Mat<Mex_GPU<double> >(mex);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}
			break;}
		}


		// Return a handle to a new C++ instance


		return;
	}

	// Check there is a second input, which should be the class instance handle
	if (nrhs < 3)
		mexErrMsgTxt("Second input should be a class instance handle.");

	// Delete
	if (!strcmp("delete", cmd)) {
		// Destroy the C++ object
		char workdirectory[180];
				mxGetString(prhs[2], workdirectory, sizeof(workdirectory));
		ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"));

		switch(configComputeArchitectureFile->getPrecision()){
		case HALF_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :
			{
				destroyObject<Mex_CPU_half >(prhs[1]);
			}
			break;

			case MEM_GPU :
			{
				destroyObject<Mex_GPU_half >(prhs[1]);
			}
			break;
			}

			break;
		}

		case FLOAT_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				destroyObject<Mex_CPU<float> >(prhs[1]);
			}
			break;

			case MEM_GPU :{
				destroyObject<Mex_GPU<float> >(prhs[1]);
			}
			break;
			}

			break;}

		case DOUBLE_GPU : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				destroyObject<Mex_CPU<double> >(prhs[1]);
			}
			break;
			case MEM_GPU :{
				destroyObject<Mex_GPU<double> >(prhs[1]);
			}
			break;
			}
			break;}
		}





		// Warn if other commands were ignored
		if (nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
		return;
	}


	char workdirectory[180];
					mxGetString(prhs[2], workdirectory, sizeof(workdirectory));
	ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"));

	switch(configComputeArchitectureFile->getPrecision()){
	case HALF_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :
		{
			Mex_CPU_half *mex_instance = convertMat2Ptr<Mex_CPU_half >(prhs[1]);
			//mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;

		case MEM_GPU :
		{
			Mex_GPU_half *mex_instance = convertMat2Ptr<Mex_GPU_half >(prhs[1]);
			//mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}

		break;
	}

	case FLOAT_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			Mex_CPU<float> *mex_instance = convertMat2Ptr<Mex_CPU<float> >(prhs[1]);
			mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;

		case MEM_GPU :{
			Mex_GPU<float> *mex_instance = convertMat2Ptr<Mex_GPU<float> >(prhs[1]);
			mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}

		break;}

	case DOUBLE_GPU : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			Mex_CPU<double> *mex_instance = convertMat2Ptr<Mex_CPU<double> >(prhs[1]);
			mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		case MEM_GPU :{
			Mex_GPU<double> *mex_instance = convertMat2Ptr<Mex_GPU<double> >(prhs[1]);
			mex_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}
		break;}
	}












	// Got here, so command not recognized
	mexErrMsgTxt("Command not recognized.");

}


/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	

	char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

	if (!strcmp("new", cmd)) {
		// Check parameters
		if (nlhs != 1)
			mexErrMsgTxt("New: One output expected.");

		ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture("./", string("configComputeArchitecture"));

		switch(configComputeArchitectureFile->getPrecision()){
		case HALF : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :
			{
				Iter3D_CPU_half* iter = new Iter3D_CPU_half("./");
				plhs[0] = convertPtr2Mat<Iter3D_CPU_half >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;

			case MEM_GPU :
			{
				Iter3D_GPU_half* iter = new Iter3D_GPU_half("./");
				plhs[0] = convertPtr2Mat<Iter3D_GPU_half >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}

			break;
		}

		case FLOAT : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				Iter3D_CPU<float>* iter = new Iter3D_CPU<float>("./");
				plhs[0] = convertPtr2Mat<Iter3D_CPU<float> >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;

			case MEM_GPU :{
				Iter3D_GPU<float>* iter = new Iter3D_GPU<float>("./");
				plhs[0] = convertPtr2Mat<Iter3D_GPU<float> >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}

			break;}

		case DOUBLE : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				Iter3D_CPU<double>* iter = new Iter3D_CPU<double>("./");
				plhs[0] = convertPtr2Mat<Iter3D_CPU<double> >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			case MEM_GPU :{
				Iter3D_GPU<double>* iter = new Iter3D_GPU<double>("./");
				plhs[0] = convertPtr2Mat<Iter3D_GPU<double> >(iter);
				//mex->doMexIter(nlhs,plhs,nrhs,prhs);
			}
			break;
			}
			break;}
		}


		// Return a handle to a new C++ instance


		return;
	}

	// Check there is a second input, which should be the class instance handle
	if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

	// Delete
	if (!strcmp("delete", cmd)) {
		// Destroy the C++ object
		ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture("./", string("configComputeArchitecture"));

		switch(configComputeArchitectureFile->getPrecision()){
		case HALF : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :
			{
				destroyObject<Iter3D_CPU_half >(prhs[1]);
			}
			break;

			case MEM_GPU :
			{
				destroyObject<Iter3D_GPU_half >(prhs[1]);
			}
			break;
			}

			break;
		}

		case FLOAT : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				destroyObject<Iter3D_CPU<float> >(prhs[1]);
			}
			break;

			case MEM_GPU :{
				destroyObject<Iter3D_GPU<float> >(prhs[1]);
			}
			break;
			}

			break;}

		case DOUBLE : {
			switch(configComputeArchitectureFile->getStorage()){
			case MEM_CPU :{
				destroyObject<Iter3D_CPU<double> >(prhs[1]);
			}
			break;
			case MEM_GPU :{
				destroyObject<Iter3D_GPU<double> >(prhs[1]);
			}
			break;
			}
			break;}
		}





		// Warn if other commands were ignored
		if (nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
		return;
	}



	ConfigComputeArchitecture *configComputeArchitectureFile=new ConfigComputeArchitecture("./", string("configComputeArchitecture"));

	switch(configComputeArchitectureFile->getPrecision()){
	case HALF : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :
		{
			Iter3D_CPU_half *iter_instance = convertMat2Ptr<Iter3D_CPU_half >(prhs[1]);
			//iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;

		case MEM_GPU :
		{
			Iter3D_GPU_half *iter_instance = convertMat2Ptr<Iter3D_GPU_half >(prhs[1]);
			//iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}

		break;
	}

	case FLOAT : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			Iter3D_CPU<float> *iter_instance = convertMat2Ptr<Iter3D_CPU<float> >(prhs[1]);
			iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;

		case MEM_GPU :{
			Iter3D_GPU<float> *iter_instance = convertMat2Ptr<Iter3D_GPU<float> >(prhs[1]);
			iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}

		break;}

	case DOUBLE : {
		switch(configComputeArchitectureFile->getStorage()){
		case MEM_CPU :{
			Iter3D_CPU<double> *iter_instance = convertMat2Ptr<Iter3D_CPU<double> >(prhs[1]);
			iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		case MEM_GPU :{
			Iter3D_GPU<double> *iter_instance = convertMat2Ptr<Iter3D_GPU<double> >(prhs[1]);
			iter_instance->doMexIter(nlhs, plhs, nrhs, prhs,(char*)cmd);
			return;
		}
		break;
		}
		break;}
	}












	// Got here, so command not recognized
	mexErrMsgTxt("Command not recognized.");

}*/



/*
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
int Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doMexIter( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

if (!strcmp("new", cmd)) {
		// Check parameters
		if (nlhs != 1)
			mexErrMsgTxt("New: One output expected.");


		Iter3D_CPU<float>* iter = new Iter3D_CPU<float>("./");

		// Return a handle to a new C++ instance
		plhs[0] = convertPtr2Mat<Iter3D_CPU<float> >(iter);

		return;
	}

	// Check there is a second input, which should be the class instance handle
	if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

	// Delete
	if (!strcmp("delete", cmd)) {
		// Destroy the C++ object
		destroyObject<Iter3D_CPU<float> >(prhs[1]);
		// Warn if other commands were ignored
		if (nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
		return;
	}

	// Get the class instance pointer from the second input
	Iter3D_CPU<float> *iter_instance = convertMat2Ptr<Iter3D_CPU<float> >(prhs[1]);

	if (!strcmp("getSinoReal", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getSinoReal: Unexpected arguments.");


		mwSize *dim_sinogram;
		dim_sinogram = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_sinogram[0] = iter_instance->getDetector()->getUDetectorPixelNb();
		dim_sinogram[1] = iter_instance->getDetector()->getVDetectorPixelNb();
		dim_sinogram[2] = iter_instance->getAcquisition()->getProjectionNb();

		plhs[0] = mxCreateNumericMatrix(iter_instance->getDetector()->getUDetectorPixelNb(),iter_instance->getDetector()->getVDetectorPixelNb()*iter_instance->getAcquisition()->getProjectionNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_sinogram, 3);
		float *SinoData=(float *) mxGetPr(plhs[0]);

		Sinogram3D_CPU<float>* realSinogram;
		realSinogram = new Sinogram3D_CPU<float>(iter_instance->getDetector()->getUDetectorPixelNb(), iter_instance->getDetector()->getVDetectorPixelNb(), iter_instance->getAcquisition()->getProjectionNb(),SinoData);
		realSinogram->loadSinogram(iter_instance->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION

		return;
	}

	if (!strcmp("CreateVolumeInit", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("CreateVolumeInit: Unexpected arguments.");

		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = iter_instance->getVolume()->getXVolumePixelNb();
		dim_volume[1] = iter_instance->getVolume()->getYVolumePixelNb();
		dim_volume[2] = iter_instance->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb()*iter_instance->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		float *ImageData=(float *) mxGetPr(plhs[0]);

		iter_instance->getVolume()->getVolumeImage()->setImageData(ImageData);

		if(iter_instance->getConfigCTFile()->getInitVolumeName().compare("none")!=0)
			iter_instance->getVolume()->loadVolume(iter_instance->getConfigCTFile()->getInitVolumeName());
		else
			iter_instance->getVolume()->setVolume(0);

		return;
	}

	if (!strcmp("getOutputDirectory", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getOutputDirectory: Unexpected arguments.");




		plhs[0] = mxCreateString(iter_instance->getOutputDirectory().c_str());

		return;
	}

	if (!strcmp("getUSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getUSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) iter_instance->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)iter_instance->getDetector()->getUDetectorPixelNb());
		return;
	}

	if (!strcmp("getVSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getVSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) iter_instance->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)iter_instance->getDetector()->getVDetectorPixelNb());
		return;
	}


	if (!strcmp("getProjectionSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getProjectionSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) iter_instance->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)iter_instance->getAcquisition()->getProjectionNb());
		return;
	}


	if (!strcmp("getXVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getXVolumePixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) iter_instance->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double) iter_instance->getVolume()->getXVolumePixelNb());
		return;
	}

	if (!strcmp("getYVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getYVolumePixelNb: Unexpected arguments.");

		plhs[0] = mxCreateDoubleScalar((double) iter_instance->getVolume()->getYVolumePixelNb());

		return;
	}


	if (!strcmp("getZVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 2)
			mexErrMsgTxt("getZVolumePixelNb: Unexpected arguments.");

		plhs[0] = mxCreateDoubleScalar((double) iter_instance->getVolume()->getZVolumePixelNb());

		return;
	}


	if (!strcmp("doBackprojection", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("doBackprojection: Unexpected arguments.");

		float *SinoData  = (float *)mxGetPr(prhs[2]);

		//Volume_CPU<float> dJ = *(iter_instance->getVolume());
		//dJ.getVolumeImage()->setImageData(ImageData);


		Sinogram3D_CPU<float>* Sinogram;
		Sinogram = new Sinogram3D_CPU<float>(iter_instance->getDetector()->getUDetectorPixelNb(), iter_instance->getDetector()->getVDetectorPixelNb(), iter_instance->getAcquisition()->getProjectionNb(),SinoData);



		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = iter_instance->getVolume()->getXVolumePixelNb();
		dim_volume[1] = iter_instance->getVolume()->getYVolumePixelNb();
		dim_volume[2] = iter_instance->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb()*iter_instance->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		float *ImageData=(float *) mxGetPr(plhs[0]);

		iter_instance->getVolume()->getVolumeImage()->setImageData(ImageData);


		iter_instance->getBackprojector()->doBackProjection(iter_instance->getVolume(),Sinogram);

		return;
	}




	if (!strcmp("doProjection", cmd)) {
		static int i=0;
		char s[255];
		// Check parameters
		i++;
		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("doProjection: Unexpected arguments.");

		float *ImageData  = (float *)mxGetPr(prhs[2]);

		//Volume_CPU<float> dJ = *(iter_instance->getVolume());
		//dJ.getVolumeImage()->setImageData(ImageData);

		iter_instance->getVolume()->getVolumeImage()->setImageData(ImageData);


		mwSize *dim_sinogram;
		dim_sinogram = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_sinogram[0] = iter_instance->getDetector()->getUDetectorPixelNb();
		dim_sinogram[1] = iter_instance->getDetector()->getVDetectorPixelNb();
		dim_sinogram[2] = iter_instance->getAcquisition()->getProjectionNb();

		plhs[0] = mxCreateNumericMatrix(iter_instance->getDetector()->getUDetectorPixelNb(),iter_instance->getDetector()->getVDetectorPixelNb()*iter_instance->getAcquisition()->getProjectionNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_sinogram, 3);
		float *SinoData=(float *) mxGetPr(plhs[0]);

		Sinogram3D_CPU<float>* estimatedSinogram;
		estimatedSinogram = new Sinogram3D_CPU<float>(iter_instance->getDetector()->getUDetectorPixelNb(), iter_instance->getDetector()->getVDetectorPixelNb(), iter_instance->getAcquisition()->getProjectionNb(),SinoData);


		iter_instance->getProjector()->doProjection(estimatedSinogram,iter_instance->getVolume());


		return;
	}

	if (!strcmp("doLaplacian", cmd)) {
			// Check parameters

			if ( nrhs < 4)
				mexErrMsgTxt("doLaplacian: Unexpected arguments.");


			float *ImageData_volume_in  = (float *)mxGetPr(prhs[2]);
			float *ImageData_volume_out  = (float *)mxGetPr(prhs[3]);


			Volume_CPU<float>* volume_in = new Volume_CPU<float>(iter_instance->getVolume()->getXVolumeSize(),iter_instance->getVolume()->getYVolumeSize(),iter_instance->getVolume()->getZVolumeSize(),iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb(),iter_instance->getVolume()->getZVolumePixelNb(),ImageData_volume_in);
			Volume_CPU<float>* volume_out = new Volume_CPU<float>(iter_instance->getVolume()->getXVolumeSize(),iter_instance->getVolume()->getYVolumeSize(),iter_instance->getVolume()->getZVolumeSize(),iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb(),iter_instance->getVolume()->getZVolumePixelNb(),ImageData_volume_out);

			float kernel_h[3] = {-1,2,-1};
			float kernel_v[3] = {-1,2,-1};
			float kernel_p[3] = {-1,2,-1};


				Convolution3D_CPU<float> convolver(kernel_h,kernel_v,kernel_p);
				convolver.doSeparableConvolution3D(volume_in,volume_out);


			return;
		}


	if (!strcmp("ApplyLaplacianRegularization_to_dJ", cmd)) {
		// Check parameters

		if ( nrhs < 9)
			mexErrMsgTxt("ApplyLaplacianRegularization_to_dJ: Unexpected arguments.");


		float *ImageData_volume  = (float *)mxGetPr(prhs[2]);
		float *ImageData_dJ  = (float *)mxGetPr(prhs[3]);
		double *JReg = (double *)mxGetPr(prhs[4]);
		double *normdJProjReg = (double *)mxGetPr(prhs[5]);
		float lambda = (float)mxGetScalar(prhs[6]);
		int totalIterationIdx = (int)mxGetScalar(prhs[7]);
		int optimalStepIterationNb = (int)mxGetScalar(prhs[8]);

		Volume_CPU<float>* dJ = new Volume_CPU<float>(iter_instance->getVolume()->getXVolumeSize(),iter_instance->getVolume()->getYVolumeSize(),iter_instance->getVolume()->getZVolumeSize(),iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb(),iter_instance->getVolume()->getZVolumePixelNb(),ImageData_dJ);
		Volume_CPU<float>* volume = new Volume_CPU<float>(iter_instance->getVolume()->getXVolumeSize(),iter_instance->getVolume()->getYVolumeSize(),iter_instance->getVolume()->getZVolumeSize(),iter_instance->getVolume()->getXVolumePixelNb(),iter_instance->getVolume()->getYVolumePixelNb(),iter_instance->getVolume()->getZVolumePixelNb(),ImageData_volume);

		if(iter_instance->getKind_Regularizer() == GG)
			iter_instance->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, JReg, normdJProjReg,lambda,totalIterationIdx,optimalStepIterationNb);
		else if(iter_instance->getKind_Regularizer() == HUBER)
			iter_instance->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, JReg, normdJProjReg,lambda,totalIterationIdx,optimalStepIterationNb);


		return;
	}


	if (!strcmp("doGradient", cmd)) {
		// Check parameters

		if ( nrhs < 4)
			mexErrMsgTxt("doGradient: Unexpected arguments.");

		float *SinoData  = (float *)mxGetPr(prhs[3]);

		Sinogram3D_CPU<float>* realSinogram;
		realSinogram = new Sinogram3D_CPU<float>(iter_instance->getDetector()->getUDetectorPixelNb(), iter_instance->getDetector()->getVDetectorPixelNb(), iter_instance->getAcquisition()->getProjectionNb(),SinoData);

		float *ImageData  = (float *)mxGetPr(prhs[2]);

		iter_instance->getVolume()->getVolumeImage()->setImageData(ImageData);

		switch(iter_instance->getKind_Gradient()){
		case SIMPLE : iter_instance->doSimpleGradient(iter_instance->getVolume(),realSinogram);break;
		case CONJUGATE : iter_instance->doConjugateGradient(iter_instance->getVolume(),realSinogram);break;
		}




		return;
	}
	}
 */
