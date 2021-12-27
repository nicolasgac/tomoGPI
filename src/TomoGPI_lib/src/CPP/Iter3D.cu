/*
 * Iter3D.cu
 *
 *      Author: gac
 */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
//#include "Iter3D.MGPUcuh"

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D(string workdirectory) : workDirectory(workdirectory),configCTFile(new ConfigCT(workdirectory, string("configCT"))),configIterationFile(new ConfigIteration(workdirectory, string("configIteration"))),configComputeArchitectureFile(new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"))),configTiffFile(new ConfigTiff<unsigned short>(workdirectory, string("configTiff"))),configProjection(new Config(workdirectory, string("configProjection"))),configProjectionScheme(new Config(workdirectory, string("configProjectionScheme"))),detector(0),fieldOfview(0),acquisition(0),proj(0),backproj(0),reg_Huber(0),reg_GG(0),volume(0),positivity(0)
{
	/********************************************************************************************************************************************/
	this->detector = &(this->getConfigCTFile()->createDetector());
	cout << "detector done" << endl;
	this->fieldOfview = &(this->getConfigCTFile()->createCylindricFOV());
	cout << "fieldOfview done" << endl;
	//Choisir le type d'acquistion uniform ou nonuniform
	if (this->getConfigCTFile()->getAcquisitionScheme().compare("Uniform") == 0)
		this->acquisition = &(this->getConfigCTFile()->createUniformAcquisition());
	/*if (this->getAcquisitionScheme().compare("NonUniform") == 0)
		this->acquisition = &(this->getConfigCTFile()->createNonUniformAcquisition());*/
	cout << "acquisition done" << endl;

	stringstream SS;
	SS << workDirectory  << "/" << this->getConfigTiffFile()->getTiffName() << ".s";
	cout << workDirectory  << endl;
	this->sinogramDirectory = SS.str();

	// geometry
	float horizontalDetectorSize = this->getConfigCTFile()->getConfigFileField<float>("horizontalDetectorSize");
	float verticalDetectorSize = this->getConfigCTFile()->getConfigFileField<float>("verticalDetectorSize");
	float focusObjectDistance = this->getConfigCTFile()->getConfigFileField<float>("FOD");
	float focusDetectorDistance = this->getConfigCTFile()->getConfigFileField<float>("FDD");
	float xMaxVolumeSize = horizontalDetectorSize*focusObjectDistance/focusDetectorDistance;
	float zMaxVolumeSize = verticalDetectorSize*(focusObjectDistance+xMaxVolumeSize/2.0)/focusDetectorDistance;

	float xVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("xVolumeSize");
	float yVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("yVolumeSize");
	unsigned long int  xVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("xVolumePixelNb");
	unsigned long int  yVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("yVolumePixelNb");
	float zVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("zVolumeSize");
	unsigned long int  zVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("zVolumePixelNb");

	
	//Regularizer<Volume_CPU,float>* regularizer;

	/*float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	this->setConvolver(new Convolution3D_CPU<float>(kernel_h,kernel_v,kernel_p));*/

	if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0){
		this->reg=HUBER;
	}else{//default
		this->reg=GG;
	}

	//if(this->getConfigIterationFile()->getRegularizationParameter() != 0)
	//{
	//if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0)
	//{
	this->setRegularizer_Huber(new R_Huber<T>(this->getConfigIterationFile()->getHuberThreshold()));

	cout << "Huber regularizer (Threshold = " << this->getConfigIterationFile()->getHuberThreshold() << ") created" << endl;
	//}
	//else if(this->getConfigIterationFile()->getPenaltyFunction().compare("GGaussian") == 0)
	//{
	this->setRegularizer_GG(new R_GG<T>(this->getConfigIterationFile()->getBeta()));

	cout << "GeneralizedGaussian regularizer created" << endl;
	//}
	//}

	if(this->getConfigIterationFile()->getGradient().compare("Simple") == 0){
		this->gradient=SIMPLE;

		cout << "Simple Gradient" << endl;
	} else if(this->getConfigIterationFile()->getGradient().compare("Conjugate") == 0)
	{
		this->gradient=CONJUGATE;

		cout << "Conjugate Gradient" << endl;
	}else if(this->getConfigIterationFile()->getGradient().compare("ProjBack") == 0) {
		this->gradient=PROJBACK;

		cout << "Proj Back" << endl;

	}
	
	this->lambda=this->getConfigIterationFile()->getRegularizationParameter();
	this->positivity=this->getConfigIterationFile()->getPositivity();
	this->noiseValue=this->getConfigIterationFile()->getAddedNoiseValue();
	this->globalIterationNb=this->getConfigIterationFile()->getGlobalIterationNb();
	this->gradientIterationNb=this->getConfigIterationFile()->getGradientIterationNb();
	this->optimalStepIterationNb=this->getConfigIterationFile()->getOptimalStepIterationNb();

	cout << "********** Start result directory creation **********" << endl;
	char tempDirectory1[180];
	char tempDirectory2[180];
	stringstream SS_out;
	time_t now = time(NULL);
	struct tm *ts = localtime(&now);
	sprintf(tempDirectory1,"%s/TomoIter/",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
#ifdef __linux__
	mkdir(tempDirectory1,S_IRWXU | S_IRWXG | S_IROTH);
#else
	mkdir(tempDirectory1);
#endif
	sprintf(tempDirectory1,"%s/TomoIter/Results",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
	strftime(tempDirectory2, sizeof(tempDirectory2), "TomoIter/Results/Result_%Y_%m_%d_%Hh%M/", ts);
	SS_out << this->getWorkDirectory()   << tempDirectory2;
	this->outputDirectory = SS_out.str();
#ifdef __linux__
	mkdir(tempDirectory1,S_IRWXU | S_IRWXG | S_IROTH);
	mkdir(this->getOutputDirectory().c_str(),S_IRWXU | S_IRWXG | S_IROTH);
#else
	mkdir(tempDirectory1);
	mkdir(this->getOutputDirectory().c_str());
#endif

	this->getConfigCTFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigCTFile()->getConfigDirectoryName() << this->getConfigCTFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigIterationFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigIterationFile()->getConfigDirectoryName() << this->getConfigIterationFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigComputeArchitectureFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigComputeArchitectureFile()->getConfigDirectoryName() << this->getConfigComputeArchitectureFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigTiffFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigTiffFile()->getConfigDirectoryName() << this->getConfigTiffFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	if(this->getConfigCTFile()->getAcquisitionScheme().compare("NonUniform")==0)
	{
		this->getConfigProjection()->copyConfigFile(this->getOutputDirectory().c_str());
		cout << this->getConfigProjection()->getConfigDirectoryName() << this->getConfigProjection()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
		this->getConfigProjectionScheme()->copyConfigFile(this->getOutputDirectory().c_str());
		cout << this->getConfigProjectionScheme()->getConfigDirectoryName() << this->getConfigProjectionScheme()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	}
	this->getConfigIterationFile()->createIterationLogFile(this->getOutputDirectory().c_str());
	cout << "********** End result directory creation **********" << endl;

}



template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::Iter3D(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : workDirectory(workdirectory),configCTFile(new ConfigCT(workdirectory, string("configCT"))),configIterationFile(new ConfigIteration(workdirectory, string("configIteration"))),configComputeArchitectureFile(configComputeArchitecture_file),configTiffFile(new ConfigTiff<unsigned short>(workdirectory, string("configTiff"))),configProjection(new Config(workdirectory, string("configProjection"))),configProjectionScheme(new Config(workdirectory, string("configProjectionScheme"))),detector(0),fieldOfview(0),acquisition(0),proj(0),backproj(0),reg_Huber(0),reg_GG(0),volume(0),positivity(0)
{
	/********************************************************************************************************************************************/



	this->detector = &(this->getConfigCTFile()->createDetector());
	cout << "detector done" << endl;
	this->fieldOfview = &(this->getConfigCTFile()->createCylindricFOV());
	cout << "fieldOfview done" << endl;
	//Choisir le type d'acquistion uniform ou nonuniform
	if (this->getConfigCTFile()->getAcquisitionScheme().compare("Uniform") == 0)
		this->acquisition = &(this->getConfigCTFile()->createUniformAcquisition());
	/*if (this->getAcquisitionScheme().compare("NonUniform") == 0)
		this->acquisition = &(this->getConfigCTFile()->createNonUniformAcquisition());*/
	cout << "acquisition done" << endl;

	stringstream SS;
	SS << workDirectory  << "/" << this->getConfigTiffFile()->getTiffName() << ".s";
	cout << workDirectory  << endl;
	this->sinogramDirectory = SS.str();

	

	// geometry
	float horizontalDetectorSize = this->getConfigCTFile()->getConfigFileField<float>("horizontalDetectorSize");
	float verticalDetectorSize = this->getConfigCTFile()->getConfigFileField<float>("verticalDetectorSize");
	float focusObjectDistance = this->getConfigCTFile()->getConfigFileField<float>("FOD");
	float focusDetectorDistance = this->getConfigCTFile()->getConfigFileField<float>("FDD");
	float xMaxVolumeSize = horizontalDetectorSize*focusObjectDistance/focusDetectorDistance;
	float zMaxVolumeSize = verticalDetectorSize*(focusObjectDistance+xMaxVolumeSize/2.0)/focusDetectorDistance;

	float xVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("xVolumeSize");
	float yVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("yVolumeSize");
	unsigned long int  xVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("xVolumePixelNb");
	unsigned long int  yVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("yVolumePixelNb");
	float zVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("zVolumeSize");
	unsigned long int  zVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("zVolumePixelNb");

	


	//Regularizer<Volume_CPU,float>* regularizer;

	/*float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	this->setConvolver(new Convolution3D_CPU<float>(kernel_h,kernel_v,kernel_p));*/

	if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0){
		this->reg=HUBER;
	}else{//default
		this->reg=GG;
	}

	//if(this->getConfigIterationFile()->getRegularizationParameter() != 0)
	//{
	//if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0)
	//{
	this->setRegularizer_Huber(new R_Huber<T>(this->getConfigIterationFile()->getHuberThreshold()));

	cout << "Huber regularizer (Threshold = " << this->getConfigIterationFile()->getHuberThreshold() << ") created" << endl;
	//}
	//else if(this->getConfigIterationFile()->getPenaltyFunction().compare("GGaussian") == 0)
	//{
	this->setRegularizer_GG(new R_GG<T>(this->getConfigIterationFile()->getBeta()));

	cout << "GeneralizedGaussian regularizer created" << endl;
	//}
	//}

	if(this->getConfigIterationFile()->getGradient().compare("Simple") == 0){
		this->gradient=SIMPLE;

		cout << "Simple Gradient" << endl;
	} else if(this->getConfigIterationFile()->getGradient().compare("Conjugate") == 0)
	{
		this->gradient=CONJUGATE;

		cout << "Conjugate Gradient" << endl;
	}else if(this->getConfigIterationFile()->getGradient().compare("ProjBack") == 0) {
		this->gradient=PROJBACK;

		cout << "Proj Back" << endl;

	}
	

	this->lambda=this->getConfigIterationFile()->getRegularizationParameter();
	this->positivity=this->getConfigIterationFile()->getPositivity();
	this->noiseValue=this->getConfigIterationFile()->getAddedNoiseValue();
	this->globalIterationNb=this->getConfigIterationFile()->getGlobalIterationNb();
	this->gradientIterationNb=this->getConfigIterationFile()->getGradientIterationNb();
	this->optimalStepIterationNb=this->getConfigIterationFile()->getOptimalStepIterationNb();

	cout << "********** Start result directory creation **********" << endl;
	char tempDirectory1[180];
	char tempDirectory2[180];
	stringstream SS_out;
	time_t now = time(NULL);
	struct tm *ts = localtime(&now);
	sprintf(tempDirectory1,"%s/TomoIter/",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
#ifdef __linux__
	mkdir(tempDirectory1,S_IRWXU | S_IRWXG | S_IROTH);
#else
	mkdir(tempDirectory1);
#endif
	sprintf(tempDirectory1,"%s/TomoIter/Results",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
	strftime(tempDirectory2, sizeof(tempDirectory2), "TomoIter/Results/Result_%Y_%m_%d_%Hh%M/", ts);
	SS_out << this->getWorkDirectory()   << tempDirectory2;
	this->outputDirectory = SS_out.str();
#ifdef __linux__
	mkdir(tempDirectory1,S_IRWXU | S_IRWXG | S_IROTH);
	mkdir(this->getOutputDirectory().c_str(),S_IRWXU | S_IRWXG | S_IROTH);
#else
	mkdir(tempDirectory1);
	mkdir(this->getOutputDirectory().c_str());
#endif

	this->getConfigCTFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigCTFile()->getConfigDirectoryName() << this->getConfigCTFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigIterationFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigIterationFile()->getConfigDirectoryName() << this->getConfigIterationFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigComputeArchitectureFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigComputeArchitectureFile()->getConfigDirectoryName() << this->getConfigComputeArchitectureFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	this->getConfigTiffFile()->copyConfigFile(this->getOutputDirectory().c_str());
	cout << this->getConfigTiffFile()->getConfigDirectoryName() << this->getConfigTiffFile()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	if(this->getConfigCTFile()->getAcquisitionScheme().compare("NonUniform")==0)
	{
		this->getConfigProjection()->copyConfigFile(this->getOutputDirectory().c_str());
		cout << this->getConfigProjection()->getConfigDirectoryName() << this->getConfigProjection()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
		this->getConfigProjectionScheme()->copyConfigFile(this->getOutputDirectory().c_str());
		cout << this->getConfigProjectionScheme()->getConfigDirectoryName() << this->getConfigProjectionScheme()->getConfigFileName() << " copied in " <<  this->getOutputDirectory() << endl;
	}
	this->getConfigIterationFile()->createIterationLogFile(this->getOutputDirectory().c_str());
	cout << "********** End result directory creation **********" << endl;

}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::~Iter3D(){
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
V<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::create_volume()
{
	V<T>* vol;
	vol=new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb());
	return vol;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
S<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::create_sinogram3D()
{
	S<T>* sino;
	sino=new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb());
	return sino;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
string Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getWorkDirectory() const
{
	return (this->workDirectory);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
string Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getSinogramDirectory() const
{
	return (this->sinogramDirectory);
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
string Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getOutputDirectory() const
{
	return (this->outputDirectory);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
ConfigIteration* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigIterationFile() const
{
	return (this->configIterationFile);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Config* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigProjection() const
{
	return (this->configProjection);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Config* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigProjectionScheme() const
{
	return (this->configProjectionScheme);
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
ConfigComputeArchitecture* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigComputeArchitectureFile() const
{
	return (this->configComputeArchitectureFile);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
ConfigCT* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigCTFile() const
{
	return (this->configCTFile);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
ConfigTiff<unsigned short>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getConfigTiffFile() const
{
	return (this->configTiffFile);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Acquisition* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getAcquisition() const
{
	return (this->acquisition);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
Detector* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getDetector() const
{
	return (this->detector);
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
CylindricFOV * Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getFieldOfview() const
{
	return (this->fieldOfview);
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
V<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getVolume() const
{
	return this->volume;
}




template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setVolume(V<T>* volume)
{
	this->volume = volume;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setBackprojector(BP<T>* backproj)
{
	this->backproj = backproj;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
BP<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getBackprojector() const
{
	return this->backproj;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setProjector(P<T>* proj)
{
	this->proj = proj;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
P<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getProjector() const
{
	return this->proj;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
kind_reg Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getKind_Regularizer() const
{
	return this->reg;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setKind_Regularizer(kind_reg reg)
{
	this->reg=reg;
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
kind_gradient Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getKind_Gradient() const
{
	return this->gradient;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setKind_Gradient(kind_gradient gradient)
{
	this->gradient=gradient;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setRegularizer_Huber(R_Huber<T>* reg)
{

	this->reg_Huber = reg;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
R_Huber<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getRegularizer_Huber() const
{
	return this->reg_Huber;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setRegularizer_GG(R_GG<T>* reg)
{

	this->reg_GG = reg;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
R_GG<T>* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getRegularizer_GG() const
{
	return this->reg_GG;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
double Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getLambda() const
{
	return this->lambda;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
char Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getPositivity() const
{
	return this->positivity;
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
double Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getNoiseValue() const
{
	return this->noiseValue;
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
unsigned int Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getGlobalIterationNb() const
{
	return this->globalIterationNb;
}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
unsigned int Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getGradientIterationNb() const
{
	return this->gradientIterationNb;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
unsigned int Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::getOptimalStepIterationNb() const
{
	return this->optimalStepIterationNb;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setLambda(double lambda)
{
	this->lambda=lambda;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setPositivity(char Positivity)
{
	this->positivity=positivity;
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setNoiseValue(double noiseValue)
{
	this->noiseValue=noiseValue;
}
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setGlobalIterationNb(unsigned int globalIterationNb)
{
	this->globalIterationNb=globalIterationNb;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setGradientIterationNb(unsigned int gradientIterationNb)
{
	this->gradientIterationNb=gradientIterationNb;
}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::setOptimalStepIterationNb(unsigned int optimalStepIterationNb)
{
	this->optimalStepIterationNb=optimalStepIterationNb;
}

#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"
//#include "Iter3D_instances_MGPU.cu"