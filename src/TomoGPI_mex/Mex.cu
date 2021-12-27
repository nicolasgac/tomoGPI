/*
 * Mex.cu
 *
 *  Created on: 15 oct. 2014
 */

#include "Mex.cuh"

template<template<typename> class I, template<typename> class V,template<typename> class S,template<typename> class C,typename T>
Mex<I,V,S,C,T>::Mex(string workdirectory)
{
	this->iter3D=new I<T>(workdirectory); // PROJECTION OBJECT CREATION
}

template<template<typename> class I, template<typename> class V,template<typename> class S,template<typename> class C,typename T>
Mex<I,V,S,C,T>::~Mex(){}


template<template<typename> class I,  template<typename> class V, template<typename> class S, template<typename> class C,typename T>
void Mex<I,V,S,C,T>::setIter3D(I<T>* iter)
{
	this->iter3D = iter;
}

template<template<typename> class I, template<typename> class V, template<typename> class S, template<typename> class C,typename T>
I<T>* Mex<I,V,S,C,T>::getIter3D() const
{
	return this->iter3D;
}

template<template<typename> class I, template<typename> class V,template<typename> class S,template<typename> class C, typename T>
void Mex<I,V,S,C,T>::doMexIter( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[],char *cmd)
{


	if (!strcmp("getVersion", cmd)) {
			// Check parameters

		printf("TomoGPI version %s\n",TOMOGPI_VERSION);
			return;
		}

	if (!strcmp("getSinoReal", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getSinoReal: Unexpected arguments.");


		mwSize *dim_sinogram;
		dim_sinogram = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_sinogram[0] = this->getIter3D()->getDetector()->getUDetectorPixelNb();
		dim_sinogram[1] = this->getIter3D()->getDetector()->getVDetectorPixelNb();
		dim_sinogram[2] = this->getIter3D()->getAcquisition()->getProjectionNb();

		plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getDetector()->getUDetectorPixelNb(),this->getIter3D()->getDetector()->getVDetectorPixelNb()*this->getIter3D()->getAcquisition()->getProjectionNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_sinogram, 3);
		T *SinoData=(T *) mxGetPr(plhs[0]);

		S<T>* realSinogram;
		realSinogram = new S<T>(this->getIter3D()->getDetector()->getUDetectorPixelNb(), this->getIter3D()->getDetector()->getVDetectorPixelNb(),this->getIter3D()->getAcquisition()->getProjectionNb(),this->getIter3D()->getCUDAArchitectureSino(),SinoData);
		realSinogram->loadSinogram(this->getIter3D()->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION

		return;
	}

	if (!strcmp("CreateVolumeInit", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("CreateVolumeInit: Unexpected arguments.");

		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = this->getIter3D()->getVolume()->getXVolumePixelNb();
		dim_volume[1] = this->getIter3D()->getVolume()->getYVolumePixelNb();
		dim_volume[2] = this->getIter3D()->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb()*this->getIter3D()->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		T *ImageData=(T *) mxGetPr(plhs[0]);

		this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);
		printf("sdfsdf\n <-- Qui debug comme ça ?");
		if(this->getIter3D()->getConfigCTFile()->getInitVolumeName().compare("none")!=0){

			this->getIter3D()->getVolume()->loadVolume(this->getIter3D()->getConfigCTFile()->getInitVolumeName());
		}
		else {
			this->getIter3D()->getVolume()->setVolume(0);
		}

		return;
	}


	if (!strcmp("CreateVolumeReal", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("CreateVolumeReal Unexpected arguments.");

		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = this->getIter3D()->getVolume()->getXVolumePixelNb();
		dim_volume[1] = this->getIter3D()->getVolume()->getYVolumePixelNb();
		dim_volume[2] = this->getIter3D()->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb()*this->getIter3D()->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		T *ImageData=(T *) mxGetPr(plhs[0]);

		this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);

		if(this->getIter3D()->getConfigCTFile()->getRealVolumeName().compare("none")!=0)
			this->getIter3D()->getVolume()->loadVolume(this->getIter3D()->getConfigCTFile()->getRealVolumeName());
		else
			this->getIter3D()->getVolume()->setVolume(0);

		return;
	}

	if (!strcmp("getOutputDirectory", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getOutputDirectory: Unexpected arguments.");




		plhs[0] = mxCreateString(this->getIter3D()->getOutputDirectory().c_str());

		return;
	}


	if (!strcmp("getWorkDirectory", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getWorkDirectory: Unexpected arguments.");




		plhs[0] = mxCreateString(this->getIter3D()->getWorkDirectory().c_str());

		return;
	}

	if (!strcmp("getDelta_un", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getDelta_un: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getDetector()->getUDetectorPixelSize());
		return;
	}

	if (!strcmp("getStorage", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getStorage: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getConfigComputeArchitectureFile()->getPrecision());
		return;
	}

	if (!strcmp("getPrecision", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getPrecision: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getConfigComputeArchitectureFile()->getStorage());
		return;
	}
	if (!strcmp("getLambda", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getLambda: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getLambda());
		return;
	}


	if (!strcmp("getPositivity", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getPositivity: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getPositivity());
		return;
	}

	if (!strcmp("setLambda", cmd)) {
		// Check parameters

		if ( nrhs < 4)
			mexErrMsgTxt("setLambda: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;
		double lambda = mxGetScalar(prhs[3]);
		this->getIter3D()->setLambda(lambda);
		return;
	}


	if (!strcmp("setPositivity", cmd)) {
		// Check parameters

		if ( nrhs < 4)
			mexErrMsgTxt("setPositivity: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;
		char positivity = (char)mxGetScalar(prhs[3]);
				this->getIter3D()->setPositivity(positivity);

		return;
	}

	if (!strcmp("getGlobalIterationNb", cmd)) {
			// Check parameters

			if (nlhs < 1 || nrhs < 3)
				mexErrMsgTxt("getGlobalIterationNb: Unexpected arguments.");

			//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
			//plhs[0] = XVolumePixelNb_ptr;

			plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getGlobalIterationNb());
			return;
		}
	if (!strcmp("getGradientIterationNb", cmd)) {
				// Check parameters

				if (nlhs < 1 || nrhs < 3)
					mexErrMsgTxt("getGradientIterationNb: Unexpected arguments.");

				//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
				//plhs[0] = XVolumePixelNb_ptr;

				plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getGradientIterationNb());
				return;
			}

	if (!strcmp("getOptimalStepIterationNb", cmd)) {
				// Check parameters

				if (nlhs < 1 || nrhs < 3)
					mexErrMsgTxt("getOptimalStepIterationNb: Unexpected arguments.");

				//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
				//plhs[0] = XVolumePixelNb_ptr;

				plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getOptimalStepIterationNb());
				return;
			}


	if (!strcmp("getUSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getUSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getDetector()->getUDetectorPixelNb());
		return;
	}

	if (!strcmp("getVSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getVSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getDetector()->getVDetectorPixelNb());
		return;
	}


	if (!strcmp("getProjectionSinogramPixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getProjectionSinogramPixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double)this->getIter3D()->getAcquisition()->getProjectionNb());
		return;
	}


	if (!strcmp("getXVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getXVolumePixelNb: Unexpected arguments.");

		//mxArray *XVolumePixelNb_ptr=mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		//plhs[0] = XVolumePixelNb_ptr;

		plhs[0] = mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getXVolumePixelNb());
		return;
	}

	if (!strcmp("getYVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getYVolumePixelNb: Unexpected arguments.");

		plhs[0] = mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getYVolumePixelNb());

		return;
	}
	if (!strcmp("getNoiseValue", cmd)) {
			// Check parameters

			if (nlhs < 1 || nrhs < 3)
				mexErrMsgTxt("getNoiseValue: Unexpected arguments.");

			plhs[0] = mxCreateDoubleScalar((double) this->getIter3D()->getNoiseValue());

			return;
		}

	if (!strcmp("getZVolumePixelNb", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 3)
			mexErrMsgTxt("getZVolumePixelNb: Unexpected arguments.");

		plhs[0] = mxCreateDoubleScalar((double) this->getIter3D()->getVolume()->getZVolumePixelNb());

		return;
	}


	if (!strcmp("doBackprojection", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 4)
			mexErrMsgTxt("doBackprojection: Unexpected arguments.");

		T *SinoData  = (T *)mxGetPr(prhs[3]);

		//Volume_CPU<float> dJ = *(this->getIter3D()->getVolume());
		//dJ.getVolumeImage()->setImageData(ImageData);


		S<T>* Sinogram;
		Sinogram = new S<T>(this->getIter3D()->getDetector()->getUDetectorPixelNb(), this->getIter3D()->getDetector()->getVDetectorPixelNb(), this->getIter3D()->getAcquisition()->getProjectionNb(),this->getIter3D()->getCUDAArchitectureSino(),SinoData);



		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = this->getIter3D()->getVolume()->getXVolumePixelNb();
		dim_volume[1] = this->getIter3D()->getVolume()->getYVolumePixelNb();
		dim_volume[2] = this->getIter3D()->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb()*this->getIter3D()->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		T *ImageData=(T *) mxGetPr(plhs[0]);

		this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);


		this->getIter3D()->getBackprojector()->doBackProjection(this->getIter3D()->getVolume(),Sinogram);

		return;
	}

	if (!strcmp("doBackprojection_FDK", cmd)) {
		// Check parameters

		if (nlhs < 1 || nrhs < 4)
			mexErrMsgTxt("doBackprojection: Unexpected arguments.");

		T *SinoData  = (T *)mxGetPr(prhs[3]);

		//Volume_CPU<float> dJ = *(this->getIter3D()->getVolume());
		//dJ.getVolumeImage()->setImageData(ImageData);


		S<T>* Sinogram;
		Sinogram = new S<T>(this->getIter3D()->getDetector()->getUDetectorPixelNb(), this->getIter3D()->getDetector()->getVDetectorPixelNb(), this->getIter3D()->getAcquisition()->getProjectionNb(),this->getIter3D()->getCUDAArchitectureSino(),SinoData);


		mwSize *dim_volume;
		dim_volume = (mwSize*) mxMalloc(3 * sizeof(mwSize));
		dim_volume[0] = this->getIter3D()->getVolume()->getXVolumePixelNb();
		dim_volume[1] = this->getIter3D()->getVolume()->getYVolumePixelNb();
		dim_volume[2] = this->getIter3D()->getVolume()->getZVolumePixelNb();

		plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb()*this->getIter3D()->getVolume()->getZVolumePixelNb(), mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(plhs[0], dim_volume, 3);
		T *ImageData=(T *) mxGetPr(plhs[0]);

		this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);

		this->getIter3D()->getBackprojector()->setFdk(1);
		this->getIter3D()->getBackprojector()->doBackProjection(this->getIter3D()->getVolume(),Sinogram);
		this->getIter3D()->getBackprojector()->setFdk(0);
		return;
	}


if (!strcmp("doProjection", cmd)) {

	char s[255];
	// Check parameters

	if (nlhs < 1 || nrhs < 4)
		mexErrMsgTxt("doProjection: Unexpected arguments.");

	T *ImageData  = (T *)mxGetPr(prhs[3]);

	//Volume_CPU<float> dJ = *(this->getIter3D()->getVolume());
	//dJ.getVolumeImage()->setImageData(ImageData);

	this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);


	mwSize *dim_sinogram;
	dim_sinogram = (mwSize*) mxMalloc(3 * sizeof(mwSize));
	dim_sinogram[0] = this->getIter3D()->getDetector()->getUDetectorPixelNb();
	dim_sinogram[1] = this->getIter3D()->getDetector()->getVDetectorPixelNb();
	dim_sinogram[2] = this->getIter3D()->getAcquisition()->getProjectionNb();

	plhs[0] = mxCreateNumericMatrix(this->getIter3D()->getDetector()->getUDetectorPixelNb(),this->getIter3D()->getDetector()->getVDetectorPixelNb()*this->getIter3D()->getAcquisition()->getProjectionNb(), mxSINGLE_CLASS, mxREAL);
	mxSetDimensions(plhs[0], dim_sinogram, 3);
	T *SinoData=(T *) mxGetPr(plhs[0]);

	S<T>* estimatedSinogram;
	estimatedSinogram = new S<T>(this->getIter3D()->getDetector()->getUDetectorPixelNb(), this->getIter3D()->getDetector()->getVDetectorPixelNb(), this->getIter3D()->getAcquisition()->getProjectionNb(),this->getIter3D()->getCUDAArchitectureSino(),SinoData);


	this->getIter3D()->getProjector()->doProjection(estimatedSinogram,this->getIter3D()->getVolume());


	return;
}


if (!strcmp("doLaplacian", cmd)) {
	// Check parameters

	if ( nrhs < 5)
		mexErrMsgTxt("doLaplacian: Unexpected arguments.");


	T *ImageData_volume_in  = (T *)mxGetPr(prhs[3]);
	T *ImageData_volume_out  = (T *)mxGetPr(prhs[4]);


	V<T>* volume_in = new V<T>(this->getIter3D()->getVolume()->getXVolumeSize(),this->getIter3D()->getVolume()->getYVolumeSize(),this->getIter3D()->getVolume()->getZVolumeSize(),this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb(),this->getIter3D()->getVolume()->getZVolumePixelNb(),this->getIter3D()->getCUDAArchitectureVolume(),ImageData_volume_in);
	V<T>* volume_out = new V<T>(this->getIter3D()->getVolume()->getXVolumeSize(),this->getIter3D()->getVolume()->getYVolumeSize(),this->getIter3D()->getVolume()->getZVolumeSize(),this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb(),this->getIter3D()->getVolume()->getZVolumePixelNb(),this->getIter3D()->getCUDAArchitectureVolume(),ImageData_volume_out);

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};


	C<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume_in,volume_out);


	return;
}


if (!strcmp("ApplyLaplacianRegularization_to_dJ", cmd)) {
	// Check parameters

	if ( nrhs < 10)
		mexErrMsgTxt("ApplyLaplacianRegularization_to_dJ: Unexpected arguments.");


	T *ImageData_volume  = (T *)mxGetPr(prhs[3]);
	T *ImageData_dJ  = (T *)mxGetPr(prhs[4]);
	double *JReg = (double *)mxGetPr(prhs[5]);
	double *normdJProjReg = (double *)mxGetPr(prhs[6]);
	float lambda = (float)mxGetScalar(prhs[7]);
	int totalIterationIdx = (int)mxGetScalar(prhs[8]);
	int optimalStepIterationNb = (int)mxGetScalar(prhs[9]);

	V<T>* dJ = new V<T>(this->getIter3D()->getVolume()->getXVolumeSize(),this->getIter3D()->getVolume()->getYVolumeSize(),this->getIter3D()->getVolume()->getZVolumeSize(),this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb(),this->getIter3D()->getVolume()->getZVolumePixelNb(),this->getIter3D()->getCUDAArchitectureVolume(),ImageData_dJ);
	V<T>* volume = new V<T>(this->getIter3D()->getVolume()->getXVolumeSize(),this->getIter3D()->getVolume()->getYVolumeSize(),this->getIter3D()->getVolume()->getZVolumeSize(),this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb(),this->getIter3D()->getVolume()->getZVolumePixelNb(),this->getIter3D()->getCUDAArchitectureVolume(),ImageData_volume);

	if(this->getIter3D()->getKind_Regularizer() == GG)
		this->getIter3D()->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, JReg, normdJProjReg,lambda,totalIterationIdx,optimalStepIterationNb);
	else if(this->getIter3D()->getKind_Regularizer() == HUBER)
		this->getIter3D()->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, JReg, normdJProjReg,lambda,totalIterationIdx,optimalStepIterationNb);


	return;
}


if (!strcmp("doGradient", cmd)) {
	// Check parameters

	if ( nrhs < 6)
		mexErrMsgTxt("doGradient: Unexpected arguments.");

	T *SinoData  = (T *)mxGetPr(prhs[4]);
	T *ImageData_volume_real  = (T *)mxGetPr(prhs[5]);

	S<T>* realSinogram;
	realSinogram = new S<T>(this->getIter3D()->getDetector()->getUDetectorPixelNb(), this->getIter3D()->getDetector()->getVDetectorPixelNb(), this->getIter3D()->getAcquisition()->getProjectionNb(),this->getIter3D()->getCUDAArchitectureSino(),SinoData);

	V<T>* realVolume;
			realVolume = new V<T>(this->getIter3D()->getVolume()->getXVolumeSize(),this->getIter3D()->getVolume()->getYVolumeSize(),this->getIter3D()->getVolume()->getZVolumeSize(),this->getIter3D()->getVolume()->getXVolumePixelNb(),this->getIter3D()->getVolume()->getYVolumePixelNb(),this->getIter3D()->getVolume()->getZVolumePixelNb(),this->getIter3D()->getCUDAArchitectureVolume(),ImageData_volume_real);;

	T *ImageData  = (T *)mxGetPr(prhs[3]);

	this->getIter3D()->getVolume()->getVolumeImage()->setImageData(ImageData);

	switch(this->getIter3D()->getKind_Gradient()){
	case SIMPLE : this->getIter3D()->doSimpleGradient(this->getIter3D()->getVolume(),realSinogram,realVolume);break;
	case CONJUGATE : this->getIter3D()->doConjugateGradient(this->getIter3D()->getVolume(),realSinogram,realVolume);break;
	}




	return;
}

	
}


//template class Iter3D<RegularSamplingProjector_CPU, VIBackProjector_CPU,int>; // 16-bit signed image
//template class Iter3D<RegularSamplingProjector_CPU, VIBackProjector_CPU,short>; // 16-bit signed image
template class Mex<Iter3D_RSVI_compute_CUDA_mem_CPU,Volume_CPU,Sinogram3D_CPU,Convolution3D_CPU,float>; // 32-bit unsigned image
template class Mex<Iter3D_RSVI_compute_CUDA_mem_CPU,Volume_CPU,Sinogram3D_CPU,Convolution3D_CPU,double>; // 64-bit signed image
template class Mex<Iter3D_RSVI_compute_CUDA_mem_GPU,Volume_GPU,Sinogram3D_GPU,Convolution3D_GPU,float>; // 32-bit unsigned image
template class Mex<Iter3D_RSVI_compute_CUDA_mem_GPU,Volume_GPU,Sinogram3D_GPU,Convolution3D_GPU,double>; // 64-bit signed image
//template class Iter3D<RegularSamplingProjector_CPU_half, VIBackProjector_CPU_half,HuberRegularizer_CPU_half,GeneralizedGaussianRegularizer_CPU_half,Convolution3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half,float>; // 64-bit signed image
//template class Iter3D<RegularSamplingProjector_GPU_half, VIBackProjector_GPU_half,HuberRegularizer_GPU_half,GeneralizedGaussianRegularizer_GPU_half,Convolution3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half,float>; // 32-bit unsigned image
//template class Projector<Volume_GPU_half, Sinogram3D_GPU_half,half>; // 64-bit signed image

/* Iter3D_CPU definition */
template <typename T>
Mex_CPU<T>::Mex_CPU(string workdirectory):Mex<Iter3D_RSVI_compute_CUDA_mem_CPU,Volume_CPU,Sinogram3D_CPU,Convolution3D_CPU,T>(workdirectory){}

template <typename T>
Mex_CPU<T>::~Mex_CPU(){}

//template class Iter3D_CPU<short>; // 32-bit unsigned image
//template class Iter3D_CPU<int>; // 64-bit signed image
template class Mex_CPU<float>; // 32-bit unsigned image
template class Mex_CPU<double>; // 64-bit signed image

template <typename T>
Mex_GPU<T>::Mex_GPU(string workdirectory) : Mex<Iter3D_RSVI_compute_CUDA_mem_GPU,Volume_GPU,Sinogram3D_GPU,Convolution3D_GPU,T>(workdirectory){}

template <typename T>
Mex_GPU<T>::~Mex_GPU(){}

template class Mex_GPU<float>; // 32-bit unsigned image
template class Mex_GPU<double>; // 64-bit signed image

template<typename I, typename V, typename S,typename C>
Mex_half<I,V,S,C>::Mex_half(){}

template<typename I, typename V, typename S, typename C>
Mex_half<I,V,S,C>::~Mex_half(){}

template class Mex_half<Iter3D_CPU_half, Volume_CPU_half,Sinogram3D_CPU_half,Convolution3D_CPU_half>; // 8-bit float image
template class Mex_half<Iter3D_GPU_half, Volume_GPU_half,Sinogram3D_GPU_half,Convolution3D_CPU_half>; // 8-bit float image

/* RegularSamplingProjector definition */
Mex_CPU_half::Mex_CPU_half(string workdirectory) : Mex_half<Iter3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half,Convolution3D_CPU_half>(){}
Mex_CPU_half::~Mex_CPU_half(){}

/* RegularSamplingProjector definition */
Mex_GPU_half::Mex_GPU_half(string workdirectory) : Mex_half<Iter3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half,Convolution3D_GPU_half>() {}
Mex_GPU_half::~Mex_GPU_half(){}
