/*
 * GeneralizedGaussianRegularizer_CPU.cu
 *
 *      Author: gac
 */


#include "GeneralizedGaussianRegularizer_CPU.cuh"

template <typename T>
GeneralizedGaussianRegularizer_CPU<T>::GeneralizedGaussianRegularizer_CPU(double beta): beta(beta){}

template <typename T>
GeneralizedGaussianRegularizer_CPU<T>::~GeneralizedGaussianRegularizer_CPU(){}

template <typename T>
double GeneralizedGaussianRegularizer_CPU<T>::getBeta() const
{
	return this->beta;
}

template <typename T>
void GeneralizedGaussianRegularizer_CPU<T>::setBeta(double beta)
{
	this->beta = beta;
}

template <typename T>
void GeneralizedGaussianRegularizer_CPU<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, unsigned int totalIterationIdx, unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

#ifdef COMPUTE_J
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
#endif
	if (this->getBeta() != 2)
	{
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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
void GeneralizedGaussianRegularizer_CPU<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
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

#endif
	if (this->getBeta() != 2)
	{
		Volume_CPU<T>* signedVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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
void GeneralizedGaussianRegularizer_CPU<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, unsigned int totalIterationIdx,unsigned  int optimalStepIterationNb) const
{
	cout << "\tGeneralized Gaussian regularization" << endl;


	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
//gradientVolume->setVolume(0);

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
	//gradientVolume->saveVolume("volume_convolue.v");


#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
				*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(this->getBeta());
	}
	cout << "\tRegularization done" << endl;
	cout << "\tjReg = " << *jReg << endl;
#endif


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
	convolver.doSeparableConvolution3D(gradientVolume,djReg);
	//gradientVolume->saveVolume("gradientvolume_convolue.v");


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
void GeneralizedGaussianRegularizer_CPU<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
#endif
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

GeneralizedGaussianRegularizer_CPU_half::GeneralizedGaussianRegularizer_CPU_half(double beta): beta(beta){}

GeneralizedGaussianRegularizer_CPU_half::~GeneralizedGaussianRegularizer_CPU_half(){}

double GeneralizedGaussianRegularizer_CPU_half::getBeta() const
{
	return this->beta;
}

void GeneralizedGaussianRegularizer_CPU_half::setBeta(double beta)
{
	this->beta = beta;
}

void GeneralizedGaussianRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};


	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture *)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
#endif
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_CPU_half* signedVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

void GeneralizedGaussianRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
#endif
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_CPU_half* signedVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

void GeneralizedGaussianRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
#endif
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_CPU_half* signedVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

void GeneralizedGaussianRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
#endif
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_CPU_half* signedVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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



#include "GeneralizedGaussianRegularizer_instances_CPU.cu"
