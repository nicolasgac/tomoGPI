/*
 * Iter3D_half.cu
 *
  *      Author: gac
 */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::Iter3D_half(string workdirectory) : workDirectory(workdirectory),configCTFile(new ConfigCT(workdirectory, string("configCT"))),configIterationFile(new ConfigIteration(workdirectory, string("configIteration"))),configComputeArchitectureFile(new ConfigComputeArchitecture(workdirectory, string("configComputeArchitecture"))),configTiffFile(new ConfigTiff<unsigned short>(workdirectory, string("configTiff"))),configProjection(new Config(workdirectory, string("configProjection"))),configProjectionScheme(new Config(workdirectory, string("configProjectionScheme"))),detector(0),fieldOfview(0),acquisition(0),cudaArchitectureSino(0), cudaArchitectureVolume(0),  cudaprojectionArchitecture(0), cudabackprojectionArchitecture(0),proj(0),backproj(0),reg_Huber(0),reg_GG(0),volume(0),positivity(0)
{
	/********************************************************************************************************************************************/

	this->detector = &(this->getConfigCTFile()->createDetector());
	cout << "detector done" << endl;
	this->fieldOfview = &(this->getConfigCTFile()->createCylindricFOV());
	cout << "fieldOfview done" << endl;
	//Choisir le type d'acquistion uniform ou nonuniform
	this->acquisition = &(this->getConfigCTFile()->createUniformAcquisition());
	cout << "acquisition done" << endl;

	cout << "half float" << endl;

	this->cudaArchitectureSino = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->cudaArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDAProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudabackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDABProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION

	stringstream SS;
	SS << workDirectory  << this->getConfigTiffFile()->getTiffName() << ".s";
	this->sinogramDirectory = SS.str();

	float xVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("xVolumeSize");
	float yVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("yVolumeSize");
	float zVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("zVolumeSize");
	float xVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("xVolumePixelNb");
	float yVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("yVolumePixelNb");
	float zVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("zVolumePixelNb");

	cout << "half floa avant set volume" << endl;

	this->setVolume(new V(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,this->cudaArchitectureVolume)); // VOLUME OBJECT CREATION
	this->setProjector(new P(this->getAcquisition(),this->getDetector(),this->getCUDAProjectionArchitecture(),this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP(this->getAcquisition(),this->getDetector(), this->getCUDABProjectionArchitecture(),this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION



	//Regularizer<Volume_CPU,float>* regularizer;

	/*float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	this->setConvolver(new Convolution3D_CPU<float>(kernel_h,kernel_v,kernel_p));*/

	if(this->getConfigIterationFile()->getRegularizationParameter() != 0)
	{
		if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0)
		{
			this->setRegularizer_Huber(new R_Huber(this->getConfigIterationFile()->getHuberThreshold()));

			cout << "Huber regularizer (Threshold = " << this->getConfigIterationFile()->getHuberThreshold() << ") created" << endl;
		}
		else if(this->getConfigIterationFile()->getPenaltyFunction().compare("GGaussian") == 0)
		{
			this->setRegularizer_GG(new R_GG(this->getConfigIterationFile()->getBeta()));

			cout << "GeneralizedGaussian regularizer created" << endl;
		}
	}

	if(this->getConfigIterationFile()->getGradient().compare("Simple") == 0){
		this->setKind_Gradient(SIMPLE);

		cout << "Simple Gradient" << endl;
	} else if(this->getConfigIterationFile()->getGradient().compare("ProjBack") == 0)
	{
		this->setKind_Gradient(PROJBACK);

		cout << "ProjBack" << endl;
	}
else if(this->getConfigIterationFile()->getGradient().compare("Conjugate") == 0)
	{
		this->setKind_Gradient(CONJUGATE);

		cout << "Conjugate Gradient" << endl;
	}

	this->setLambda(this->getConfigIterationFile()->getRegularizationParameter());
	this->setPositivity(this->getConfigIterationFile()->getPositivity());
	this->setNoiseValue(this->getConfigIterationFile()->getAddedNoiseValue());
	this->setGlobalIterationNb(this->getConfigIterationFile()->getGlobalIterationNb());
	this->setGradientIterationNb(this->getConfigIterationFile()->getGradientIterationNb());
	this->setOptimalStepIterationNb(this->getConfigIterationFile()->getOptimalStepIterationNb());

	cout << "********** Start result directory creation **********" << endl;
	char tempDirectory1[180];
	char tempDirectory2[180];
	stringstream SS_out;
	time_t now = time(NULL);
	struct tm *ts = localtime(&now);
	sprintf(tempDirectory1,"%s/TomoIter/",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
#ifdef __linux__
	mkdir(tempDirectory1, S_IRWXU | S_IRWXG | S_IROTH);
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


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::Iter3D_half(string workdirectory, ConfigComputeArchitecture* ConfigComputeArchitecture_file) : workDirectory(workdirectory),configCTFile(new ConfigCT(workdirectory, string("configCT"))),configIterationFile(new ConfigIteration(workdirectory, string("configIteration"))),configComputeArchitectureFile(ConfigComputeArchitecture_file),configTiffFile(new ConfigTiff<unsigned short>(workdirectory, string("configTiff"))),configProjection(new Config(workdirectory, string("configProjection"))),configProjectionScheme(new Config(workdirectory, string("configProjectionScheme"))),detector(0),fieldOfview(0),acquisition(0),cudaArchitectureSino(0), cudaArchitectureVolume(0),  cudaprojectionArchitecture(0), cudabackprojectionArchitecture(0),proj(0),backproj(0),reg_Huber(0),reg_GG(0),volume(0),positivity(0)
{
	/********************************************************************************************************************************************/



	this->detector = &(this->getConfigCTFile()->createDetector());
	cout << "detector done" << endl;
	this->fieldOfview = &(this->getConfigCTFile()->createCylindricFOV());
	cout << "fieldOfview done" << endl;
	//Choisir le type d'acquistion uniform ou nonuniform
	this->acquisition = &(this->getConfigCTFile()->createUniformAcquisition());
	cout << "acquisition done" << endl;

	cout << "half float" << endl;

	this->cudaArchitectureSino = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureSino(this->getDetector(),this->getAcquisition()));
	this->cudaArchitectureVolume = &(this->getConfigComputeArchitectureFile()->createCUDAArchitectureVolume(this->getFieldOfview())); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDAProjectionArchitecture()); // PROJECTION ARCHITECTURE OBJECT CREATION
	this->cudabackprojectionArchitecture = &(this->getConfigComputeArchitectureFile()->createCUDABProjectionArchitecture(this->getFieldOfview())); // BACKPROJECTION ARCHITECTURE OBJECT CREATION
	this->cudaArchitectureSino->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_sino());
	this->cudaArchitectureVolume->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_vol());
	this->cudaprojectionArchitecture->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_proj());
	this->cudabackprojectionArchitecture->setComputingUnitNb(this->getConfigComputeArchitectureFile()->getGpuNb_back());
this->cudaprojectionArchitecture->setProjectionStreamsNb(this->getConfigComputeArchitectureFile()->getprojectionStreamsNb());
this->cudabackprojectionArchitecture->setBProjectionStreamsNb(this->getConfigComputeArchitectureFile()->getbackprojectionStreamsNb());


	stringstream SS;
	SS << workDirectory  << this->getConfigTiffFile()->getTiffName() << ".s";
	this->sinogramDirectory = SS.str();

	float xVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("xVolumeSize");
	float yVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("yVolumeSize");
	float zVolumeSize = this->getConfigCTFile()->getConfigFileField<float>("zVolumeSize");
	float xVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("xVolumePixelNb");
	float yVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("yVolumePixelNb");
	float zVolumePixelNb = this->getConfigCTFile()->getConfigFileField<int>("zVolumePixelNb");

	cout << "half floa avant set volume" << endl;

	this->setVolume(new V(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,this->cudaArchitectureVolume)); // VOLUME OBJECT CREATION
	this->setProjector(new P(this->getAcquisition(),this->getDetector(),this->getCUDAProjectionArchitecture(),this->getVolume())); // PROJECTION OBJECT CREATION
	this->setBackprojector(new BP(this->getAcquisition(),this->getDetector(), this->getCUDABProjectionArchitecture(),this->getVolume(),0)); // BACK PROJECTION OBJECT CREATION



	//Regularizer<Volume_CPU,float>* regularizer;

	/*float kernel_h[3] = {-1,2,-1};
	float kernel_v[3] = {-1,2,-1};
	float kernel_p[3] = {-1,2,-1};

	this->setConvolver(new Convolution3D_CPU<float>(kernel_h,kernel_v,kernel_p));*/

	if(this->getConfigIterationFile()->getRegularizationParameter() != 0)
	{
		if(this->getConfigIterationFile()->getPenaltyFunction().compare("Huber") == 0)
		{
			this->setRegularizer_Huber(new R_Huber(this->getConfigIterationFile()->getHuberThreshold()));

			cout << "Huber regularizer (Threshold = " << this->getConfigIterationFile()->getHuberThreshold() << ") created" << endl;
		}
		else if(this->getConfigIterationFile()->getPenaltyFunction().compare("GGaussian") == 0)
		{
			this->setRegularizer_GG(new R_GG(this->getConfigIterationFile()->getBeta()));

			cout << "GeneralizedGaussian regularizer created" << endl;
		}
	}

	if(this->getConfigIterationFile()->getGradient().compare("Simple") == 0){
		this->setKind_Gradient(SIMPLE);

		cout << "Simple Gradient" << endl;
	} else if(this->getConfigIterationFile()->getGradient().compare("ProjBack") == 0)
	{
		this->setKind_Gradient(PROJBACK);

		cout << "ProjBack" << endl;
	}
else if(this->getConfigIterationFile()->getGradient().compare("Conjugate") == 0)
	{
		this->setKind_Gradient(CONJUGATE);

		cout << "Conjugate Gradient" << endl;
	}


	this->setLambda(this->getConfigIterationFile()->getRegularizationParameter());
	this->setPositivity(this->getConfigIterationFile()->getPositivity());
	this->setNoiseValue(this->getConfigIterationFile()->getAddedNoiseValue());
	this->setGlobalIterationNb(this->getConfigIterationFile()->getGlobalIterationNb());
	this->setGradientIterationNb(this->getConfigIterationFile()->getGradientIterationNb());
	this->setOptimalStepIterationNb(this->getConfigIterationFile()->getOptimalStepIterationNb());

	cout << "********** Start result directory creation **********" << endl;
	char tempDirectory1[180];
	char tempDirectory2[180];
	stringstream SS_out;
	time_t now = time(NULL);
	struct tm *ts = localtime(&now);
	sprintf(tempDirectory1,"%s/TomoIter/",this->getWorkDirectory().c_str(),this->getAcquisition()->getProjectionNb());
#ifdef __linux__
	mkdir(tempDirectory1, S_IRWXU | S_IRWXG | S_IROTH);
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


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::~Iter3D_half(){}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
string Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getWorkDirectory() const
{
	return (this->workDirectory);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
string Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getSinogramDirectory() const
{
	return (this->sinogramDirectory);
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
string Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getOutputDirectory() const
{
	return (this->outputDirectory);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
ConfigIteration* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigIterationFile() const
{
	return (this->configIterationFile);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Config* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigProjection() const
{
	return (this->configProjection);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Config* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigProjectionScheme() const
{
	return (this->configProjectionScheme);
}
template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
ConfigComputeArchitecture* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigComputeArchitectureFile() const
{
	return (this->configComputeArchitectureFile);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
ConfigCT* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigCTFile() const
{
	return (this->configCTFile);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
ConfigTiff<unsigned short>* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getConfigTiffFile() const
{
	return (this->configTiffFile);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Acquisition* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getAcquisition() const
{
	return (this->acquisition);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
Detector* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getDetector() const
{
	return (this->detector);
}
template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setNoiseValue(double noiseValue)
{
	this->noiseValue=noiseValue;
}
template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
double Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getNoiseValue() const
{
	return this->noiseValue;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
CylindricFOV * Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getFieldOfview() const
{
	return (this->fieldOfview);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
CUDAArchitecture* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getCUDAArchitectureSino() const
{
	return (this->cudaArchitectureSino);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
CUDAArchitecture* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getCUDAArchitectureVolume() const
{
	return (this->cudaArchitectureVolume);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
CUDAProjectionArchitecture* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getCUDAProjectionArchitecture() const
{
	return (this->cudaprojectionArchitecture);
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
CUDABProjectionArchitecture* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getCUDABProjectionArchitecture() const
{
	return (this->cudabackprojectionArchitecture);
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
V* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getVolume() const
{
	return this->volume;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setVolume(V* volume)
{
	this->volume = volume;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setBackprojector(BP* backproj)
{
	this->backproj = backproj;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
BP* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getBackprojector() const
{
	return this->backproj;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setProjector(P* proj)
{
	this->proj = proj;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
P* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getProjector() const
{
	return this->proj;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
kind_reg Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getKind_Regularizer() const
{
	return this->reg;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setKind_Regularizer(kind_reg reg)
{
	this->reg=reg;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setPositivity(char Positivity)
{
	this->positivity=positivity;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
kind_gradient Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getKind_Gradient() const
{
	return this->gradient;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setKind_Gradient(kind_gradient gradient)
{
	this->gradient=gradient;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setRegularizer_Huber(R_Huber* reg)
{
	this->reg=HUBER;
	this->reg_Huber = reg;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
R_Huber* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getRegularizer_Huber() const
{
	return this->reg_Huber;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setRegularizer_GG(R_GG* reg)
{
	this->reg=GG;
	this->reg_GG = reg;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
R_GG* Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getRegularizer_GG() const
{
	return this->reg_GG;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
double Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getLambda() const
{
	return this->lambda;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
unsigned int Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getGlobalIterationNb() const
{
	return this->globalIterationNb;
}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
unsigned int Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getGradientIterationNb() const
{
	return this->gradientIterationNb;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
unsigned int Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getOptimalStepIterationNb() const
{
	return this->optimalStepIterationNb;
}
template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
char Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::getPositivity() const
{
	return this->positivity;
}
template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setLambda(double lambda)
{
	this->lambda=lambda;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setGlobalIterationNb(unsigned int globalIterationNb)
{
	this->globalIterationNb=globalIterationNb;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setGradientIterationNb(unsigned int gradientIterationNb)
{
	this->gradientIterationNb=gradientIterationNb;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::setOptimalStepIterationNb(unsigned int optimalStepIterationNb)
{
	this->optimalStepIterationNb=optimalStepIterationNb;
}

template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::doSimpleGradient(V* volume,S* realSinogram,V* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	V* dJ = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	//dJ->setVolume(0);
#ifdef COMPUTE_EAM
	V* volume_temp = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	volume_temp->setVolume(0);
#endif
	S* estimatedSinogram;
	estimatedSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);



	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double alpha_min = -999999999999;
	double beta=0.0;
	double eam=0.0;

	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;


	// Gradient descent
	cout << "Start gradient descent" << endl;
	error = cudaEventRecord(start, NULL);
	for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
	{
		//			gettimeofday(&startc, NULL);
		cout << "Gradient iteration n° " << gradientIterationIdx << endl;

		/* Projection */
		//volume->saveVolume(this->getOutputDirectory() + "avant_proj"+ std::to_string(gradientIterationIdx) + ".v");

		this->getProjector()->doProjection(estimatedSinogram,volume);
		cout << "\tProjection done "  << endl;
		//estimatedSinogram->saveSinogram(this->getOutputDirectory() + "avant_diffsino"+ std::to_string(gradientIterationIdx) + ".s");

		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

		//J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		//cout << "\tJ_MC = " << J_MC << endl;

		/* BackProjection */
		//dJ->saveVolume(this->getOutputDirectory() + "avant_BP"+ std::to_string(gradientIterationIdx) + ".v");
		dJ->setVolume(0);
		//estimatedSinogram->saveSinogram(this->getOutputDirectory() + "avant_BP"+ std::to_string(gradientIterationIdx) + ".s");
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);

		cout << "\tBackprojection done "  << endl;

		//delete estimatedSinogram;
		//dJ->saveVolume(this->getOutputDirectory() + "apres_BP"+ std::to_string(gradientIterationIdx) + ".v");
		dJ->scalarVolume(-2.0);
		//dJ->saveVolume(this->getOutputDirectory() + "apres_scalarvolume"+ std::to_string(gradientIterationIdx) + ".v");

		/* Regularization */
		if(this->getLambda() != 0){
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
		}
		//dJ->saveVolume(this->getOutputDirectory() + "apres_reg"+ std::to_string(gradientIterationIdx) + ".v");


		/* Step Computation */
		if(gradientIterationIdx < this->getOptimalStepIterationNb())
		{
			cout << "\tStart Descent Step Computation" << endl;
			this->getProjector()->doProjection(estimatedSinogram,dJ);
			normdJ = dJ->getVolumeL2Norm();
			//normdJ = dJ->getVolumeL2Norm<float>();
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

			cout << "\tnormdJ = " << normdJ << " normdJProj = "  << normdJProj << "normdJProjReg = "  << normdJProjReg << endl;

			alpha=-1.0*(normdJ)/(normdJProj);
			alpha_min = fmax(alpha,alpha_min);
			cout << "\tEnd Descent Step Computation (alpha = " << alpha << ")" << endl;
		}
		else
		{
			alpha = alpha_min;
			cout << "\talpha = " << alpha << ")" << endl;
		}
		//			delete estimatedSinogram;

		cout << "\tStart Volume Updating" << endl;
		//volume->saveVolume(this->getOutputDirectory() + "Volume_avant_update"+ std::to_string(gradientIterationIdx) + ".v");
		this->getProjector()->updateVolume(volume,dJ,alpha,this->getPositivity());
		//volume->saveVolume(this->getOutputDirectory() + "apres_update"+ std::to_string(gradientIterationIdx) + ".v");

		cout << "\tEnd Volume Updating" << endl;

#ifdef COMPUTE_MIDDLE_SLICE
		volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
		//volume->saveVolume(this->getOutputDirectory() + "Volume"+ std::to_string(gradientIterationIdx) + ".v");
#ifdef COMPUTE_EAM
		volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
		volume_temp->diffVolume(realVolume);
		eam=volume_temp->getVolumeL1Norm();
#endif

#ifdef COMPUTE_J
		J = J_MC + lambda*JReg;
		cout << "\tJ = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(gradientIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif
		//			gettimeofday(&end, NULL);
		//			seconds  = end.tv_sec  - startc.tv_sec;
		//			useconds = end.tv_usec - startc.tv_usec;
		//			msecTotal = useconds;
		//			msecTotal/=1000;
		//			msecTotal+=seconds*1000;
		//			cout << "Time iter = " << msecTotal <<  " msec" << endl;
	}

	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);


	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	volume->saveVolume(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;

}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::doWeightedGradient(V* volume,S* realSinogram,V* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	V* dJ = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
#ifdef COMPUTE_EAM
	V* volume_temp = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	volume_temp->setVolume(0);
#endif
	S* estimatedSinogram;
	estimatedSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);



	S* weightedSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	stringstream weightedSinogramFile;
	weightedSinogramFile << this->sinogramDirectory << "/weightedSinogram.s";
	weightedSinogram->loadSinogram(weightedSinogramFile.str() );
	cout << "Weight sinogram initialized" << endl;

	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double alpha_min = -999999999999;
	double beta=0.0;
	double eam=0.0;

	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;


	// Gradient descent
	cout << "Start gradient descent" << endl;
	error = cudaEventRecord(start, NULL);
	for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
	{
		//			gettimeofday(&startc, NULL);
		cout << "Gradient iteration n° " << gradientIterationIdx << endl;

		/* Projection */
		this->getProjector()->doProjection(estimatedSinogram,volume);
		cout << "Projection done "  << endl;

		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

		J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		cout << "J_MC = " << J_MC << endl;

		estimatedSinogram->multSinogram(weightedSinogram);

		/* BackProjection */
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
		cout << "Backprojection done "  << endl;

		//delete estimatedSinogram;
		dJ->scalarVolume(-2.0);

		/* Regularization */
		if(this->getLambda() != 0){
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
		}

		/* Step Computation */
		if(gradientIterationIdx < this->getOptimalStepIterationNb())
		{
			cout << "Start Descent Step Computation" << endl;
			this->getProjector()->doProjection(estimatedSinogram,dJ);

			normdJ = dJ->getVolumeL2Norm();
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

			cout << "normdJ = " << normdJ << "normdJProj = "  << normdJProj << "normdJProjReg = "  << normdJProjReg << endl;

			alpha=-1.0*(normdJ)/(normdJProj);
			alpha_min = fmax(alpha,alpha_min);
			cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;
		}
		else
		{
			alpha = alpha_min;
			cout << "alpha = " << alpha << ")" << endl;
		}
		//			delete estimatedSinogram;

		cout << "Start Volume Updating" << endl;

		this->getProjector()->updateVolume(volume,dJ,alpha,this->getPositivity());

		cout << "End Volume Updating" << endl;
#ifdef COMPUTE_MIDDLE_SLICE
		volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
#ifdef COMPUTE_EAM
		volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
		volume_temp->diffVolume(realVolume);
		eam=volume_temp->getVolumeL1Norm();
#endif
#ifdef COMPUTE_J
		J = J_MC + lambda*JReg;
		cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(gradientIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif
		//			gettimeofday(&end, NULL);
		//			seconds  = end.tv_sec  - startc.tv_sec;
		//			useconds = end.tv_usec - startc.tv_usec;
		//			msecTotal = useconds;
		//			msecTotal/=1000;
		//			msecTotal+=seconds*1000;
		//			cout << "Time iter = " << msecTotal <<  " msec" << endl;
	}




	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	volume->saveVolume(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	delete weightedSinogram;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;

}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
void Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::doConjugateGradient(V* volume,S* realSinogram,V* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	V* dJ = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
#ifdef COMPUTE_EAM
	V* volume_temp = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	volume_temp->setVolume(0);
#endif

	S* estimatedSinogram;
	estimatedSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);

	V* olddJ = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	V* p = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);


	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normolddJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;

	double beta = 0.0,eam=0.0;

	unsigned int totalIterationIdx = 0;
	unsigned int globalIterationIdx;
	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;

	error = cudaEventRecord(start, NULL);
	for(globalIterationIdx = 1; globalIterationIdx <= this->getGlobalIterationNb(); globalIterationIdx++)
	{
		cout << "Global iteration n° " << globalIterationIdx << endl;

		// Conjugate Gradient descent
		cout << "Start conjugate gradient descent" << endl;

		// Init Conjugate Gradient descent
		cout << "Initialization conjugate gradient descent" << endl;

		gradientIterationIdx = 1;
		cout << "Gradient iteration n° " << gradientIterationIdx << endl;
		totalIterationIdx = (globalIterationIdx-1)*this->getGradientIterationNb()+gradientIterationIdx;
		cout << "Total iteration n° " << totalIterationIdx << endl;

		this->getProjector()->doProjection(estimatedSinogram, volume);
		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);
		J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		cout << "J_MC = " << J_MC << endl;
		/* BackProjection */
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
		dJ->scalarVolume(-2.0);

		/* Regularization */
		if(this->getLambda() != 0){
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),totalIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),totalIterationIdx,this->getOptimalStepIterationNb());
		}

		*olddJ=*p;

		/* Step Computation */

		cout << "Start Descent Step Computation" << endl;
		this->getProjector()->doProjection(estimatedSinogram,p);
		normdJ = p->getVolumeL2Norm();
		normolddJ = normdJ;
		normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

		alpha=-1.0*(normdJ)/(normdJProj);
		cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;


		cout << "Start Volume Updating" << endl;
		this->getProjector()->updateVolume(volume,p,alpha,this->getPositivity());
		cout << "End Volume Updating" << endl;
		this->getProjector()->getVolume()->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
		J = J_MC + this->getLambda()*JReg;
		cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(totalIterationIdx,J,J_MC,JReg,alpha,beta,eam);
		cout << "End Initialization conjugate gradient descent" << endl;


		for (gradientIterationIdx = 2 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
		{

			cout << "Gradient iteration n° " << gradientIterationIdx << endl;
			totalIterationIdx = (globalIterationIdx-1)*this->getGradientIterationNb()+gradientIterationIdx;
			cout << "Total iteration n° " << totalIterationIdx << endl;
			/* Projection */
			this->getProjector()->doProjection(estimatedSinogram, volume);
			estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

			J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
			cout << "J_MC = " << J_MC << endl;


			/* BackProjection */
			this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
			dJ->scalarVolume(-2.0);


			/* Regularization */
			if(this->getLambda() != 0){
				if(this->getKind_Regularizer() == GG)
					this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, p, olddJ,&JReg, &normdJProjReg, &normdJ, &normolddJ, &beta,this->getLambda());
				else if(this->getKind_Regularizer() == HUBER)
					this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ,p, olddJ, &JReg, &normdJProjReg,&normdJ, &normolddJ, &beta,this->getLambda());
			}


			cout << "Estimated sinogram initialized" << endl;

			/* Step Computation */

			cout << "Start Descent Step Computation" << endl;

			this->getProjector()->doProjection(estimatedSinogram,p);
			normdJ = p->scalarProductVolume(dJ);
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

			alpha=-1.0*(normdJ)/(normdJProj);
			cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;

			cout << "Start Volume Updating" << endl;
			this->getProjector()->updateVolume(volume,p,alpha,this->getPositivity());
			cout << "End Volume Updating" << endl;
#ifdef COMPUTE_MIDDLE_SLICE
			volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
#ifdef COMPUTE_EAM
			volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
			volume_temp->diffVolume(realVolume);
			eam=volume_temp->getVolumeL1Norm();
#endif
#ifdef COMPUTE_J
			J = J_MC + lambda*JReg;
			cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
			this->getConfigIterationFile()->updateLogFile(totalIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif



		}


	}


	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);

	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	volume->saveVolume(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	//delete olddJ;
	//delete p;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;

}


template<typename P, typename BP, typename R_Huber, typename R_GG,typename C, typename V, typename S>
int Iter3D_half<P,BP,R_Huber,R_GG,C,V,S>::doMainIter( int argc, char** argv)
{
	printf("TomoGPI version %s\n",TOMOGPI_VERSION);

	V* volume = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
	if(this->getConfigCTFile()->getInitVolumeName().compare("none")!=0)
		volume->loadVolume(this->getConfigCTFile()->getInitVolumeName());

	V* volume_real;
	if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){
		volume_real= new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
		volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
	}



	char *op_string = NULL;

	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	high_resolution_clock::time_point t0,t1;

	if (getCmdLineArgumentString(argc, (const char**)argv, "op", &op_string)){

		if (strcmp(op_string,"proj") == 0){

			S* sino = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			/*cout << "********** WARM UP**************" << endl;
							this->getProjector()->doProjection(sino,volume_real);
							cout << "********** END WARM UP**********" << endl;*/

			t0 = high_resolution_clock::now();
			this->getProjector()->doProjection(sino,volume_real);
			t1 = high_resolution_clock::now();




			sino->saveSinogram("proj_half.s");

		} else if (strcmp(op_string,"reg_GG") == 0){
			V* dJ = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
			dJ->setVolume(0);
			unsigned int gradientIterationIdx=1;
			double JReg = 0.0;
			double normdJProjReg = 0.0;
			cout << "hello" << endl;
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());

			dJ->saveVolume("GG_test.v");

		} else if (strcmp(op_string,"laplacien") == 0){
			float kernel_h[3] = {-1,2,-1};
			float kernel_v[3] = {-1,2,-1};
			float kernel_p[3] = {-1,2,-1};

			V* volume_out = new V(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
			volume_out->setVolume(0);

			C convolver(kernel_h,kernel_v,kernel_p);
			convolver.doSeparableConvolution3D(volume_real,volume_out);

			volume_out->saveVolume("laplacien_test.v");

		} else if (strcmp(op_string,"back") == 0){





			S* realSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION

			/*cout << "********** WARM UP **************" << endl;
			this->getBackprojector()->doBackProjection(volume,realSinogram);
			cout << "********** END WARM UP **********" << endl;*/


			t0 = high_resolution_clock::now();
			this->getBackprojector()->doBackProjection(volume,realSinogram);
			t1 = high_resolution_clock::now();

			volume->saveVolume("back_half.v");

		}
	}
	else {

		S*realSinogram = new S(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
		realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION


		t0 = high_resolution_clock::now();
		switch(this->getKind_Gradient()){
		case SIMPLE : this->doSimpleGradient(volume,realSinogram,volume_real);
		cout << "Simple Gradient" << endl;
		break;
		case CONJUGATE : this->doConjugateGradient(volume,realSinogram,volume_real);
		cout << "Conjugate Gradient" << endl;
		break;
		}
		t1 = high_resolution_clock::now();

	}


	/*if(this->getConfigCTFile()->getBuildRealSinogramFromTiff())
	{Sinogram3D_CPU<float> *realSinogram_tiff;//(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	realSinogram_tiff = &(this->getConfigTiffFile()->createNormalizedMeanNegativeLogSinogram());
	cout << "createNormalizedNegativeLogSinogram done" << endl;
	realSinogram_tiff->saveSinogram(this->getSinogramDirectory());
	delete realSinogram_tiff;
	cout << "Real sinogram from tiff created" << endl;
	}*/


	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout <<"Time computation :"  << total_ms.count() << "ms\n";

	return 0;

}


/* RegularSamplingProjector definition */
Iter3D_CPU_half::Iter3D_CPU_half(string workdirectory) : Iter3D_half<RegularSamplingProjector_CPU_half,VIBackProjector_CPU_half,HuberRegularizer_CPU_half,GeneralizedGaussianRegularizer_CPU_half,Convolution3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half>(workdirectory){}


Iter3D_CPU_half::Iter3D_CPU_half(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D_half<RegularSamplingProjector_CPU_half,VIBackProjector_CPU_half,HuberRegularizer_CPU_half,GeneralizedGaussianRegularizer_CPU_half,Convolution3D_CPU_half,Volume_CPU_half,Sinogram3D_CPU_half>(workdirectory,configComputeArchitecture_file){}

Iter3D_CPU_half::~Iter3D_CPU_half(){}

/* RegularSamplingProjector definition */
Iter3D_GPU_half::Iter3D_GPU_half(string workdirectory) : Iter3D_half<RegularSamplingProjector_GPU_half,VIBackProjector_GPU_half,HuberRegularizer_GPU_half,GeneralizedGaussianRegularizer_GPU_half,Convolution3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half>(workdirectory) {}


Iter3D_GPU_half::Iter3D_GPU_half(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file) : Iter3D_half<RegularSamplingProjector_GPU_half,VIBackProjector_GPU_half,HuberRegularizer_GPU_half,GeneralizedGaussianRegularizer_GPU_half,Convolution3D_GPU_half,Volume_GPU_half,Sinogram3D_GPU_half>(workdirectory,configComputeArchitecture_file) {}
Iter3D_GPU_half::~Iter3D_GPU_half(){}

#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"