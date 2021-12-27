/*
 * GeneralizedGaussianRegularizer_CPU.cpp
 *
 *      Author: gac
 */


#include "GeneralizedGaussianRegularizer_CPU.cuh"

template <typename T>
GeneralizedGaussianRegularizer<T>::GeneralizedGaussianRegularizer(double beta): beta(beta){}

template <typename T>
GeneralizedGaussianRegularizer<T>::~GeneralizedGaussianRegularizer(){}

template <typename T>
double GeneralizedGaussianRegularizer<T>::getBeta() const
{
	return this->beta;
}

template <typename T>
void GeneralizedGaussianRegularizer<T>::setBeta(double beta)
{
	this->beta = beta;
}

template <typename T>
void GeneralizedGaussianRegularizer<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

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
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(gradientVolume,djReg,4);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		convolver.doSeparableConvolution3D(dJ,gradientVolume,4);

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
void GeneralizedGaussianRegularizer<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

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
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(gradientVolume,djReg,4);

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

	convolver.doSeparableConvolution3D(p,gradientVolume,4);

	if (beta == 2)
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
void GeneralizedGaussianRegularizer<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	//	gradientVolume->saveVolume("/espace/boulay/gradientCPU.v");

	if (this->getBeta() == 2)
	{
		//		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(gradientVolume,djReg,4);

	//	djReg->saveVolume("/espace/boulay/dJRegCPU.v");
	//		dJ->saveVolume("/espace/boulay/dJCPU.v");
	dJ->addVolume(djReg,this->getBeta()*lambda);
	//	dJ->saveVolume("/espace/boulay/dJCPU2.v");

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{
		convolver.doSeparableConvolution3D(dJ,gradientVolume,4);

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
void GeneralizedGaussianRegularizer<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	if (this->getBeta() == 2)
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
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(gradientVolume,djReg,4);

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

	convolver.doSeparableConvolution3D(p,gradientVolume,4);

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

template class GeneralizedGaussianRegularizer<float>; // 32-bit real Generalized Gaussian Regularizer
template class GeneralizedGaussianRegularizer<double>; // 64-bit real Generalized Gaussian Regularizer

