/*
 * HuberRegularizer_CPU_half.cu
 *
 *      Author: gac
 */


#include "HuberRegularizer_CPU_half.cuh"

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

void HuberRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

void HuberRegularizer_CPU_half::getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;


	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

	Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

void HuberRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

//		gradientVolume->saveVolume("/espace/boulay/gradient1.v");

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	convolver.doSeparableConvolution3D(derivativeHuberVolume,djReg,4);

	dJ->addVolume(djReg,lambda);

	delete djReg;

	delete derivativeHuberVolume;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

void HuberRegularizer_CPU_half::getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const
{
	cout << "********** Start calcul of Huber regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	Volume_CPU_half* gradientVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());

	Convolution3D_CPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume,4);

	*jReg=gradientVolume->getVolumeHuberNorm(this->getHuberThreshold());

	cout << "********** End calcul of Huber regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	Volume_CPU_half* derivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->derivativeHuberFunction(gradientVolume, derivativeHuberVolume);

	Volume_CPU_half* djReg = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
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

	Volume_CPU_half* secondDerivativeHuberVolume = new Volume_CPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb());
	this->secondDerivativeHuberFunction(gradientVolume, secondDerivativeHuberVolume);
	secondDerivativeHuberVolume->multVolume(p);
	convolver.doSeparableConvolution3D(secondDerivativeHuberVolume,gradientVolume,4);
	convolver.doSeparableConvolution3D(gradientVolume,secondDerivativeHuberVolume,4);
	secondDerivativeHuberVolume->multVolume(p);
	*normdJProjReg = secondDerivativeHuberVolume->getVolumeL1Norm();

	delete secondDerivativeHuberVolume;
	delete gradientVolume;
}

