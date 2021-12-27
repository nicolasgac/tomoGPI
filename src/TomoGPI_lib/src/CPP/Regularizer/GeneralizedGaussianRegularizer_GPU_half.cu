/*
 * GeneralizedGaussianRegularizer_GPU_half.cu
 *
 *      Author: gac
 */


#include "GeneralizedGaussianRegularizer_GPU_half.cuh"

GeneralizedGaussianRegularizer_GPU_half::GeneralizedGaussianRegularizer_GPU_half(double beta): beta(beta){}

GeneralizedGaussianRegularizer_GPU_half::~GeneralizedGaussianRegularizer_GPU_half(){}

double GeneralizedGaussianRegularizer_GPU_half::getBeta() const
{
	return this->beta;
}

void GeneralizedGaussianRegularizer_GPU_half::setBeta(double beta)
{
	this->beta = beta;
}

void GeneralizedGaussianRegularizer_GPU_half::getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;


	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = ((Image3D_GPU<float>*)volume->getVolumeImage())->getCUDAArchitecture();

	Volume_GPU_half* gradientVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU_half* signedVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU_half* djReg = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		convolver.doSeparableConvolution3D(dJ,gradientVolume);

		if (this->getBeta() == 2)
		{
			*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm<float>();
		}
		else
		{
			*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
		}
	}

	delete gradientVolume;
}

void GeneralizedGaussianRegularizer_GPU_half::getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,0,1};
	float kernel_v[3] = {1,2,1};
	float kernel_p[3] = {1,2,1};

	CUDAArchitecture* gpuArch = ((Image3D_GPU<float>*)volume->getVolumeImage())->getCUDAArchitecture();

	Volume_GPU_half* gradientVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU_half* signedVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU_half* djReg = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume<float>(olddJ);
	*betaCG = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm<float>();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *betaCG << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*betaCG, dJ);
	cout << "End p Updating" << endl;

	convolver.doSeparableConvolution3D(p,gradientVolume);

	if (this->getBeta() == 2)
	{
		*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
	}

	delete gradientVolume;
}

void GeneralizedGaussianRegularizer_GPU_half::getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, int totalIterationIdx, int optimalStepIterationNb) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = ((Image3D_GPU<float>*)volume->getVolumeImage())->getCUDAArchitecture();

	Volume_GPU_half* gradientVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		//		*jReg=gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU_half* signedVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU_half* djReg = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	if(totalIterationIdx < optimalStepIterationNb)
	{

		convolver.doSeparableConvolution3D(dJ,gradientVolume);

		if (this->getBeta() == 2)
		{
			*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm<float>();
		}
		else
		{
			*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
		}
	}

	delete gradientVolume;
}

void GeneralizedGaussianRegularizer_GPU_half::getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const
{
	cout << "********** Start calcul of Generalized Gaussian regularization criterion jReg **********" << endl;

	float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	CUDAArchitecture* gpuArch = ((Image3D_GPU<float>*)volume->getVolumeImage())->getCUDAArchitecture();

	Volume_GPU_half* gradientVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);

	Convolution3D_GPU_half convolver(kernel_h,kernel_v,kernel_p);
	convolver.doSeparableConvolution3D(volume,gradientVolume);

	if (this->getBeta() == 2)
	{
		*jReg=gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*jReg=gradientVolume->getVolumeLpNorm(beta);
	}
	cout << "********** End calcul of Generalized Gaussian regularization criterion jReg **********" << endl;
	cout << "jReg = " << *jReg << endl;

	if (this->getBeta() != 2)
	{
		Volume_GPU_half* signedVolume = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
		volume->getVolumeSign(signedVolume);
		volume->getVolumeAbsPow(gradientVolume,beta-1);
		gradientVolume->multVolume(signedVolume);
		delete signedVolume;
	}

	Volume_GPU_half* djReg = new Volume_GPU_half(volume->getXVolumeSize(),volume->getYVolumeSize(),volume->getZVolumeSize(),volume->getXVolumePixelNb(),volume->getYVolumePixelNb(),volume->getZVolumePixelNb(),gpuArch);
	convolver.doSeparableConvolution3D(gradientVolume,djReg);

	dJ->addVolume(djReg,this->getBeta()*lambda);

	delete djReg;

	olddJ->diffVolume(dJ);
	*normdJ = dJ->scalarProductVolume<float>(olddJ);
	*betaCG = -1.0*(*normdJ)/(*normolddJ);
	*normdJ = dJ->getVolumeL2Norm<float>();
	*normolddJ = *normdJ;
	olddJ = dJ;
	cout << "Beta = " << *betaCG << endl;

	cout << "Start p Updating" << endl;
	p->diffVolume(*betaCG, dJ);
	cout << "End p Updating" << endl;

	convolver.doSeparableConvolution3D(p,gradientVolume);

	if (this->getBeta() == 2)
	{
		*normdJProjReg = 2.0*gradientVolume->getVolumeL2Norm<float>();
	}
	else
	{
		*normdJProjReg = this->getBeta()*gradientVolume->getVolumeLpNorm(beta);
	}

	delete gradientVolume;
}
