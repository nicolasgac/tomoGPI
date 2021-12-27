/*
 * HuberRegularizer_CPU.cpp
 *
 *      Author: gac
 */


#include "HuberRegularizer_CPU.cuh"

template <typename T>
HuberRegularizer<T>::HuberRegularizer(double huberThreshold): huberThreshold(huberThreshold){}

template <typename T>
HuberRegularizer<T>::~HuberRegularizer(){}

template <typename T>
double HuberRegularizer<T>::getHuberThreshold() const
{
	return this->huberThreshold;
}

template <typename T>
void HuberRegularizer<T>::setHuberThreshold(double huberThreshold)
{
	this->huberThreshold = huberThreshold;
}

template <typename T>
void HuberRegularizer<T>::derivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* derivativeHuberVolume) const
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
void HuberRegularizer<T>::secondDerivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* secondDerivativeHuberVolume) const
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
void HuberRegularizer<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
		convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
		secondDerivativeHuberVolume->multVolume(dJ);
		*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

		delete secondDerivativeHuberVolume;
	}

	delete gradientVolume;
}

template <typename T>
void HuberRegularizer<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

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

	Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

template <typename T>
void HuberRegularizer<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	//	gradientVolume->saveVolume("/espace/boulay/gradient3.v");

	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{
		Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
		secondDerivativeHuberVolume->multVolume(dJ);
		convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
		convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
		secondDerivativeHuberVolume->multVolume(dJ);
		*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

		delete secondDerivativeHuberVolume;
	}

	delete gradientVolume;
}

template <typename T>
void HuberRegularizer<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

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

	Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

template class HuberRegularizer<float>; // 32-bit real Huber Regularizer
template class HuberRegularizer<double>; // 64-bit real Huber Regularizer
