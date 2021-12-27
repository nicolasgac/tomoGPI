/*
 * ConfigIteration.cpp
 *
  *      Author: gac
 */

#include "ConfigIteration.hpp"

ConfigIteration::ConfigIteration() : Config() , addedNoiseValue(0), globalIterationNb(0), optimalStepIterationNb(0), gradientIterationNb(0), regularizationParameter(0), logFileDirectory(""), gradient("Simple"),penaltyFunction(""),huberThreshold(0),beta(0),positivity(0), gammaPotts(0), segmentationIterationNb(0), m0(0), v0(0), alpha0(0), beta0(0), alpha_epsilon0(0), beta_epsilon0(0), number_classes(0), tol(0),tol_seg(0),potts_mgi_mgm("MGI"){}
ConfigIteration::ConfigIteration(string configDirectoryName, string configFileName) : Config(configDirectoryName, configFileName)
{
	this->addedNoiseValue = this->getConfigFileField<int>("addedNoiseValue");
	this->globalIterationNb = this->getConfigFileField<int>("globalIterationNb");
	this->gradientIterationNb = this->getConfigFileField<int>("gradientIterationNb");
	this->optimalStepIterationNb = this->getConfigFileField<int>("optimalStepIterationNb");
	this->regularizationParameter = this->getConfigFileField<double>("regularizationParameter");
	this->logFileDirectory = configDirectoryName;
	this->penaltyFunction = this->getConfigFileField<string>("penaltyFunction");
	this->gradient = this->getConfigFileField<string>("gradient");
	this->positivity = this->getConfigFileField<int>("positivity");
	this->huberThreshold = this->getConfigFileField<double>("huberThreshold");
	this->beta = this->getConfigFileField<double>("beta");
	this->gammaPotts=this->getConfigFileField<double>("PottsCoefficient");
	this->segmentationIterationNb=this->getConfigFileField<int>("segmentationIterationNb");
	this->m0=this->getConfigFileField<double>("m0GaussianPriorMeansClasses");
	this->v0=this->getConfigFileField<double>("v0GaussianPriorMeansClasses");
	this->alpha0=this->getConfigFileField<double>("alpha0InverseGammaPriorVariancesClasses");
	this->beta0=this->getConfigFileField<double>("beta0InverseGammaPriorVariancesClasses");
	this->alpha_epsilon0=this->getConfigFileField<double>("alphaNoiseInverseGammaPriorVariancesNoise");
	this->beta_epsilon0=this->getConfigFileField<double>("betaNoiseInverseGammaPriorVariancesNoise");
	this->number_classes=this->getConfigFileField<int>("numberOfClasses");
	this->tol=this->getConfigFileField<double>("tolerance");
	this->tol_seg=this->getConfigFileField<double>("toleranceSegmentation");
	this->potts_mgi_mgm=this->getConfigFileField<string>("PottsMGIMGM");
}

ConfigIteration::~ConfigIteration(){}

unsigned short ConfigIteration::getAddedNoiseValue()
{
	return this->addedNoiseValue;
}

unsigned short ConfigIteration::getGlobalIterationNb()
{
	return this->globalIterationNb;
}

unsigned short ConfigIteration::getGradientIterationNb()
{
	return this->gradientIterationNb;
}

unsigned short ConfigIteration::getPositivity()
{
	return this->positivity;
}
unsigned short ConfigIteration::getOptimalStepIterationNb()
{
	return this->optimalStepIterationNb;
}

double ConfigIteration::getRegularizationParameter()
{
	return this->regularizationParameter;
}

string ConfigIteration::getLogFileDirectory()
{
	return this->logFileDirectory;
}

string ConfigIteration::getGradient()
{
	return this->gradient;
}

string ConfigIteration::getPenaltyFunction()
{
	return this->penaltyFunction;
}

double ConfigIteration::getHuberThreshold()
{
	return this->huberThreshold;
}

double ConfigIteration::getBeta()
{
	return this->beta;
}

double ConfigIteration::getGammaPotts()
{
	return this->gammaPotts;
}

unsigned short ConfigIteration::getSegmentationIterationNb()
{
	return this->segmentationIterationNb;
}

double ConfigIteration::getM0()
{
	return this->m0;
}

double ConfigIteration::getV0()
{
	return this->v0;
}

double ConfigIteration::getAlpha0()
{
	return this->alpha0;
}

double ConfigIteration::getBeta0()
{
	return this->beta0;
}

double ConfigIteration::getAlphaEpsilon0()
{
	return this->alpha_epsilon0;
}

double ConfigIteration::getBetaEpsilon0()
{
	return this->beta_epsilon0;
}

int ConfigIteration::getNumberClasses()
{
	return this->number_classes;
}

double ConfigIteration::getTol()
{
	return this->tol;
}

string ConfigIteration::getPottsMGIMGM()
{
	return this->potts_mgi_mgm;
}

double ConfigIteration::getTolSegmentation()
{
	return this->tol_seg;
}


void ConfigIteration::setAddedNoiseValue(unsigned short addedNoiseValue)
{
	this->addedNoiseValue = addedNoiseValue;
}

void ConfigIteration::setGlobalIterationNb(unsigned short globalIterationNb)
{
	this->globalIterationNb = globalIterationNb;
}

void ConfigIteration::setPositivity(char positivity)
{
	this->positivity = positivity;
}
void ConfigIteration::setGradientIterationNb(unsigned short gradientIterationNb)
{
	this->gradientIterationNb = gradientIterationNb;
}

void ConfigIteration::setOptimalStepIterationNb(unsigned short gradientIterationNb)
{
	this->optimalStepIterationNb = optimalStepIterationNb;
}

void ConfigIteration::setRegularizationParameter(double regularizationParameter)
{
	this->regularizationParameter = regularizationParameter;
}

void ConfigIteration::setLogFileDirectory(string logFileDirectory)
{
	this->logFileDirectory = logFileDirectory;
}

void ConfigIteration::setPenaltyFunction(string penaltyFunction)
{
	this->penaltyFunction=penaltyFunction;
}

void ConfigIteration::setGradient(string gradient)
{
	this->gradient=gradient;
}

void ConfigIteration::setHuberThreshold(double huberThreshold)
{
	this->huberThreshold = huberThreshold;
}

void ConfigIteration::setBeta(double beta)
{
	this->beta=beta;
}

void ConfigIteration::setGammaPotts(double gammaPotts)
{
	this->gammaPotts=gammaPotts;
}

void ConfigIteration::setSegmentationIterationNb(unsigned short segmentationIterationNb)
{
	this->segmentationIterationNb=segmentationIterationNb;
}

void ConfigIteration::setM0(double m0)
{
	this->m0=m0;
}

void ConfigIteration::setV0(double v0)
{
	this->v0=v0;
}

void ConfigIteration::setAlpha0(double alpha0)
{
	this->alpha0=alpha0;
}

void ConfigIteration::setBeta0(double beta0)
{
	this->beta0=beta0;
}

void ConfigIteration::setAlphaEpsilon0(double alpha_epsilon0)
{
	this->alpha_epsilon0=alpha_epsilon0;
}

void ConfigIteration::setBetaEpsilon0(double beta_epsilon0)
{
	this->beta_epsilon0=beta_epsilon0;
}

void ConfigIteration::setNumberClasses(int number_classes)
{
	this->beta_epsilon0=number_classes;
}

void ConfigIteration::setTol(double tol)
{
	this->tol=tol;
}

void ConfigIteration::setPottsMGIMGM(string potts_mgi_mgm)
{
	this->potts_mgi_mgm=potts_mgi_mgm;
}

void ConfigIteration::setTolSegmentation(double tol_seg)
{
	this->tol_seg=tol_seg;
}

void ConfigIteration::createIterationLogFile(string destinationDirectory)
{
	ofstream iterationLogFile;
	this->setLogFileDirectory(destinationDirectory);
	string directory = this->getLogFileDirectory();
	iterationLogFile.open((directory+"Iteration.dat").c_str(), ios::out);
	int i;

	if(iterationLogFile.is_open())
	{
		iterationLogFile << setw(20) << left << "IterNb" << setw(20) << left << "J" << setw(20) << left << "J_MC" << setw(20) << left << "JReg" << setw(20) << left << "lambda" << setw(20) << left << "alpha" << setw(20) << left << "beta" <<  endl;
		iterationLogFile.close();
		string name = "chmod 774 ";
		i=system((name + (directory+"Iteration.dat").c_str()).c_str());
	}
	else
	{
		cout << "Unable to create file " << directory+"Iteration.dat" << endl;
		exit(EXIT_FAILURE);
	}
}

void ConfigIteration::updateLogFile(unsigned short iter, double J, double J_MC, double J_Reg, double alpha, double beta, double eam)
{
	ofstream iterationLogFile;
	string directory = this->getLogFileDirectory();
	iterationLogFile.open((directory+"Iteration.dat").c_str(), ios::out |ios::app);

	if(iterationLogFile.is_open())
	{
		iterationLogFile << setw(20) << left << iter << setw(20) << left << J << setw(20) << left << J_MC << setw(20) << left << J_Reg << setw(20) << left << getRegularizationParameter() << setw(20) << left << abs(alpha) << setw(20) << left << beta << setw(20) << left << eam << endl;
		iterationLogFile.close();
	}
	else
	{
		cout << "Unable to open file " << directory+"Iteration.dat" << endl;
		exit(EXIT_FAILURE);
	}
}
