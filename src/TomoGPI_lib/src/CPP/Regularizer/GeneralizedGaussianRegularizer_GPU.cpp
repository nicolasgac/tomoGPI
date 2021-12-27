/*
 * GeneralizedGaussianRegularizer_GPU.cpp
 *
 *      Author: gac
 */


#include "GeneralizedGaussianRegularizer_GPU.cuh"

template <typename T>
GeneralizedGaussianRegularizer_GPU<T>::GeneralizedGaussianRegularizer_GPU(double beta): beta(beta){}

template <typename T>
GeneralizedGaussianRegularizer_GPU<T>::~GeneralizedGaussianRegularizer_GPU(){}

template <typename T>
double GeneralizedGaussianRegularizer_GPU<T>::getBeta() const
{
	return this->beta;
}

template <typename T>
void GeneralizedGaussianRegularizer_GPU<T>::setBeta(double beta)
{
	this->beta = beta;
}

template <typename T>
void GeneralizedGaussianRegularizer_GPU<T>::getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU<T>* signedVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		convolver.doSeparableConvolution3D(dJ,gradientVolume);

		if (this->getBeta() == 2)
		{
			*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm();
		}
		else
		{
			*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
		}
	}

	delete gradientVolume;
}

template <typename T>
void GeneralizedGaussianRegularizer_GPU<T>::getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU<T>* signedVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume(olddJ);
	*betaCG = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *betaCG << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*betaCG, dJ);
	cout << "End p Updating" << endl;

	convolver.doSeparableConvolution3D(p,gradientVolume);

	if (this->getBeta() == 2)
	{
		*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
	}

	delete gradientVolume;
}

template <typename T>
void GeneralizedGaussianRegularizer_GPU<T>::getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	//	gradientVolume->saveVolume("/espace/boulay/gradientGPU.v");

	if (this->getBeta() == 2.0)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2.0)
	{
		Volume_GPU<T>* signedVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	//	djReg->saveVolume("/espace/boulay/dJRegGPU.v");
	//	dJ->saveVolume("/espace/boulay/dJGPU.v");
	dJ->addVolume(djReg,this->getBeta()*lambda);
	//	dJ->saveVolume("/espace/boulay/dJGPU2.v");

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{
		convolver.doSeparableConvolution3D(dJ,gradientVolume);

		if (this->getBeta() == 2.0)
		{
			*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm();
		}
		else
		{
			*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
		}
	}

	delete gradientVolume;
}

template <typename T>
void GeneralizedGaussianRegularizer_GPU<T>::getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (beta == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU<T>* signedVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume(olddJ);
	*betaCG = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *betaCG << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*betaCG, dJ);
	cout << "End p Updating" << endl;

	convolver.doSeparableConvolution3D(p,gradientVolume);

	if (this->getBeta() == 2)
	{
		*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
	}

	delete gradientVolume;
}

template class GeneralizedGaussianRegularizer_GPU<float>; // 32-bit real Generalized Gaussian Regularizer
template class GeneralizedGaussianRegularizer_GPU<double>; // 64-bit real Generalized Gaussian Regularizer

