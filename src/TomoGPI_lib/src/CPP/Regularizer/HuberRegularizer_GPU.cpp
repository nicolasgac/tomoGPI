/*
 * HuberRegularizer_GPU.cpp
 *
 *      Author: gac
 */

#include "HuberRegularizer_GPU.cuh"

template <typename T>
HuberRegularizer_GPU<T>::HuberRegularizer_GPU(double huberThreshold): huberThreshold(huberThreshold){}

template <typename T>
HuberRegularizer_GPU<T>::~HuberRegularizer_GPU(){}

template <typename T>
double HuberRegularizer_GPU<T>::getHuberThreshold() const
{
	return this->huberThreshold;
}

template <typename T>
void HuberRegularizer_GPU<T>::setHuberThreshold(double huberThreshold)
{
	this->huberThreshold = huberThreshold;
}

template <typename T>
void HuberRegularizer_GPU<T>::derivativeHuberFunction(Volume_GPU<T>* volume, Volume_GPU<T>* derivativeHuberVolume) const
{
	if(volume->isSameSize(derivativeHuberVolume))
	{
		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = volume->getXVolumePixelNb();
		yNb = volume->getYVolumePixelNb();
		zNb = volume->getZVolumePixelNb();
		float tmp = 0;
		double huberThres = this->getHuberThreshold();

		T* volumeData = volume->getVolumeData();
		T* derivativeHuberVolumeData = derivativeHuberVolume->getVolumeData();

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					tmp = volumeData[x+y*xNb+z*xNb*yNb];
					if(tmp <= -1.0*huberThres)
						derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = -2.0*huberThres;
					else if (fabs(tmp) < huberThres)
						derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = 2.0*tmp;
					else if (tmp >= huberThres)
						derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = 2.0*huberThres;
					else
						derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = tmp;
				}
	}
	else
	{
		cout << "Volumes must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void HuberRegularizer_GPU<T>::secondDerivativeHuberFunction(Volume_GPU<T>* volume, Volume_GPU<T>* secondDerivativeHuberVolume) const
{
	if(volume->isSameSize(secondDerivativeHuberVolume))
	{
		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = volume->getXVolumePixelNb();
		yNb = volume->getYVolumePixelNb();
		zNb = volume->getZVolumePixelNb();
		float tmp = 0;
		double huberThres = this->getHuberThreshold();

		T* volumeData = volume->getVolumeData();
		T* secondDerivativeHuberVolumeData = secondDerivativeHuberVolume->getVolumeData();

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					tmp = volumeData[x+y*xNb+z*xNb*yNb];
					if(fabs(tmp) > huberThres)
						secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = 0.0;
					else if (fabs(tmp) <= huberThres)
						secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = 1.0;
					else
						secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb] = tmp;
				}
	}
	else
	{
		cout << "Volumes must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void HuberRegularizer_GPU<T>::getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_GPU<T>* derivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_GPU<T>* secondDerivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
		convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

		delete secondDerivativeHuberVolume;
	}

	delete gradientVolume;
}

template <typename T>
void HuberRegularizer_GPU<T>::getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_GPU<T>* derivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume(olddJ);
	*beta = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *beta << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*beta, dJ);
	cout << "End p Updating" << endl;

	Volume_GPU<T>* secondDerivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

template <typename T>
void HuberRegularizer_GPU<T>::getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	//	gradientVolume->saveVolume("/espace/boulay/gradient1.v");

	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_GPU<T>* derivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_GPU<T>* secondDerivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
		convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

		delete secondDerivativeHuberVolume;
	}
	delete gradientVolume;
}

template <typename T>
void HuberRegularizer_GPU<T>::getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = volume->getVolumeImage()->getCUDAArchitecture();

	Volume_GPU<T>* gradientVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_GPU<T>* derivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_GPU<T>* djReg = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume(olddJ);
	*beta = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *beta << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*beta, dJ);
	cout << "End p Updating" << endl;

	Volume_GPU<T>* secondDerivativeHuberVolume = new Volume_GPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

template class HuberRegularizer_GPU<float>; // 32-bit real Huber Regularizer
template class HuberRegularizer_GPU<double>; // 64-bit real Huber Regularizer

