/*
 * HuberRegularizer_CPU.cu
 *
 *      Author: gac
 */


#include "HuberRegularizer_CPU.cuh"
#include "GPUConstant.cuh"
//#include "HuberRegularizer_kernel_half.cuh"

template <typename T>
HuberRegularizer_CPU<T>::HuberRegularizer_CPU(double huberThreshold): huberThreshold(huberThreshold){}

template <typename T>
HuberRegularizer_CPU<T>::~HuberRegularizer_CPU(){}

template <typename T>
double HuberRegularizer_CPU<T>::getHuberThreshold() const
{
	return this->huberThreshold;
}

template <typename T>
void HuberRegularizer_CPU<T>::setHuberThreshold(double huberThreshold)
{
	this->huberThreshold = huberThreshold;
}

template <typename T>
void HuberRegularizer_CPU<T>::derivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* derivativeHuberVolume) const
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
void HuberRegularizer_CPU<T>::secondDerivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* secondDerivativeHuberVolume) const
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
void HuberRegularizer_CPU<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	//cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
	cout << "/tjReg = " << *jReg << endl;
#endif
	//cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;


	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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
void HuberRegularizer_CPU<T>::getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{


	T kernel_h[3] = {-1,0,1};
	T kernel_v[3] = {1,2,1};
	T kernel_p[3] = {1,2,1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
	cout << "/tjReg = " << *jReg << endl;
#endif

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

	Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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
void HuberRegularizer_CPU<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	//	gradientVolume->saveVolume("/espace/boulay/gradient3.v");
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{
		Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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
void HuberRegularizer_CPU<T>::getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	T kernel_h[3] = {-1,2,-1};
	T kernel_v[3] = {-1,2,-1};
	T kernel_p[3] = {-1,2,-1};

	Volume_CPU<T>* gradientVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU<T> convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(huberThreshold);
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU<T>* derivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU<T>* djReg = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

	Volume_CPU<T>* secondDerivativeHuberVolume = new Volume_CPU<T>(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}


HuberRegularizer_CPU_half::HuberRegularizer_CPU_half(double huberThreshold): huberThreshold(huberThreshold){}
HuberRegularizer_CPU_half::~HuberRegularizer_CPU_half(){}

double HuberRegularizer_CPU_half::getHuberThreshold() const
{
	return this->huberThreshold;
}

void HuberRegularizer_CPU_half::setHuberThreshold(double huberThreshold)
{
	this->huberThreshold = huberThreshold;
}

void HuberRegularizer_CPU_half::derivativeHuberFunction(Volume_CPU_half* volume, Volume_CPU_half* derivativeHuberVolume) const
{
	if(volume->isSameSize(derivativeHuberVolume))
	{
		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = volume->getXVolumePixelNb();
		yNb = volume->getYVolumePixelNb();
		zNb = volume->getZVolumePixelNb();
		float tmp = 0;
		float tmp2 = 0;
		double huberThres = this->getHuberThreshold();

		half* volumeData = volume->getVolumeData();
		half* derivativeHuberVolumeData =  derivativeHuberVolume->getVolumeData();

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp,&volumeData[x+y*xNb+z*xNb*yNb],1);
					if(tmp <= -1.0*huberThres)
					{
						tmp2=-2.0*(float)huberThres;
						singles2halfp(&derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
					else if (fabs(tmp) < huberThres)
					{
						tmp2=2.0*tmp;
						singles2halfp(&derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
					else if (tmp >= huberThres)
					{
						tmp2=2.0*(float)huberThres;
						singles2halfp(&derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
					else
						singles2halfp(&derivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp,1);
				}
	}
	else
	{
		cout << "Volumes must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void HuberRegularizer_CPU_half::secondDerivativeHuberFunction(Volume_CPU_half* volume, Volume_CPU_half* secondDerivativeHuberVolume) const
{
	if(volume->isSameSize(secondDerivativeHuberVolume))
	{
		unsigned long int x,y,z,xNb,yNb,zNb;
		xNb = volume->getXVolumePixelNb();
		yNb = volume->getYVolumePixelNb();
		zNb = volume->getZVolumePixelNb();
		float tmp = 0;
		float tmp2 = 0;
		double huberThres = this->getHuberThreshold();

		half* volumeData = volume->getVolumeData();
		half* secondDerivativeHuberVolumeData = secondDerivativeHuberVolume->getVolumeData();

		for (z=0;z<zNb;z++)
			for (y=0;y<yNb;y++)
				for (x=0;x<xNb;x++)
				{
					halfp2singles(&tmp,&volumeData[x+y*xNb+z*xNb*yNb],1);
					if(fabs(tmp) > huberThres)
					{
						tmp2=0.0;
						singles2halfp(&secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
					else if (fabs(tmp) <= huberThres)
					{
						tmp2=1.0;
						singles2halfp(&secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
					else
					{
						tmp2=tmp;
						singles2halfp(&secondDerivativeHuberVolumeData[x+y*xNb+z*xNb*yNb],&tmp2,1);
					}
				}
	}
	else
	{
		cout << "Volumes must have the same size" << endl;
		exit(EXIT_FAILURE);
	}
}

void HuberRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

void HuberRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;


	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

	Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

void HuberRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

//		gradientVolume->saveVolume("/espace/boulay/gradient1.v");
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

void HuberRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);
#ifdef COMPUTE_J
	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());
#endif
	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
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

	Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),(CUDAArchitecture*)NULL);
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

#include "HuberRegularizer_instances_CPU.cu"