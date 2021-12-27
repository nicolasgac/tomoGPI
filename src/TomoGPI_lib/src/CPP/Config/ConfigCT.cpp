/*
 * ConfigCT.cpp
 *
 *      Author: gac
 */

#include "ConfigCT.hpp"

ConfigCT::ConfigCT() : Config(), buildRealSinogramFromTiff(0){}

ConfigCT::ConfigCT(string configDirectoryName, string configFileName) : Config(configDirectoryName, configFileName)
{
	acquisitionScheme = this->getConfigFileField<string>("acquisitionScheme");
	initVolumeName = this->getConfigFileField<string>("initVolumeName");
	realVolumeName = this->getConfigFileField<string>("realVolumeName");
	buildRealSinogramFromTiff = this->getConfigFileField<int>("buildRealSinogramFromTiff");
}

ConfigCT::~ConfigCT(){}

string ConfigCT::getAcquisitionScheme()
{
	return this->acquisitionScheme;
}

string ConfigCT::getInitVolumeName()
{
	return this->initVolumeName;
}

string ConfigCT::getRealVolumeName()
{
	return this->realVolumeName;
}

int ConfigCT::getBuildRealSinogramFromTiff()
{
	return this->buildRealSinogramFromTiff;
}

/*
template <typename T>
		Volume_CPU<T> & ConfigCT::createVolumeCPU()
		{
			float xVolumeSize = this->getConfigFileField<float>("xVolumeSize");
			float yVolumeSize = this->getConfigFileField<float>("yVolumeSize");
			float zVolumeSize = this->getConfigFileField<float>("zVolumeSize");
			unsigned long int xVolumePixelNb = this->getConfigFileField<int>("xVolumePixelNb");
			unsigned long int yVolumePixelNb = this->getConfigFileField<int>("yVolumePixelNb");
			unsigned long int zVolumePixelNb = this->getConfigFileField<int>("zVolumePixelNb");
			Volume_CPU<T>* volume = new Volume_CPU<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb);

			return *volume;
		}
	template <typename T>
		Volume_CPU<T> & ConfigCT::createVolumeCPU(T* dataImage)
		{
			float xVolumeSize = this->getConfigFileField<float>("xVolumeSize");
			float yVolumeSize = this->getConfigFileField<float>("yVolumeSize");
			float zVolumeSize = this->getConfigFileField<float>("zVolumeSize");
			unsigned long int xVolumePixelNb = this->getConfigFileField<int>("xVolumePixelNb");
			unsigned long int yVolumePixelNb = this->getConfigFileField<int>("yVolumePixelNb");
			unsigned long int zVolumePixelNb = this->getConfigFileField<int>("zVolumePixelNb");
			Volume_CPU<T>* volume = new Volume_CPU<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,dataImage);

			return *volume;
		}



	Volume_CPU_half  ConfigCT::createVolumeCPU_half()
	{
		float xVolumeSize = this->getConfigFileField<float>("xVolumeSize");
		float yVolumeSize = this->getConfigFileField<float>("yVolumeSize");
		float zVolumeSize = this->getConfigFileField<float>("zVolumeSize");
		unsigned long int xVolumePixelNb = this->getConfigFileField<int>("xVolumePixelNb");
		unsigned long int yVolumePixelNb = this->getConfigFileField<int>("yVolumePixelNb");
		unsigned long int zVolumePixelNb = this->getConfigFileField<int>("zVolumePixelNb");
		Volume_CPU_half* volume = new Volume_CPU_half(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb);

		return *volume;
	}

	template <typename T>
	Volume_GPU<T>  createVolumeGPUcreateVolumeGPU(CUDAArchitecture* cudaArchitecture)
	{
		float xVolumeSize = this->getConfigFileField<float>("xVolumeSize");
		float yVolumeSize = this->getConfigFileField<float>("yVolumeSize");
		float zVolumeSize = this->getConfigFileField<float>("zVolumeSize");
		unsigned long int xVolumePixelNb = this->getConfigFileField<int>("xVolumePixelNb");
		unsigned long int yVolumePixelNb = this->getConfigFileField<int>("yVolumePixelNb");
		unsigned long int zVolumePixelNb = this->getConfigFileField<int>("zVolumePixelNb");
		Volume_GPU<T>* volume = new Volume_GPU<T>(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture);

		return *volume;
	}

	Volume_GPU_half  createVolumeGPUcreateVolumeGPU_half(CUDAArchitecture* cudaArchitecture)
	{
		float xVolumeSize = this->getConfigFileField<float>("xVolumeSize");
		float yVolumeSize = this->getConfigFileField<float>("yVolumeSize");
		float zVolumeSize = this->getConfigFileField<float>("zVolumeSize");
		unsigned long int xVolumePixelNb = this->getConfigFileField<int>("xVolumePixelNb");
		unsigned long int yVolumePixelNb = this->getConfigFileField<int>("yVolumePixelNb");
		unsigned long int zVolumePixelNb = this->getConfigFileField<int>("zVolumePixelNb");
		Volume_GPU_half* volume = new Volume_GPU_half(xVolumeSize,yVolumeSize,zVolumeSize,xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture);

		return *volume;
	}*/
UniformAcquisition & ConfigCT::createUniformAcquisition()
{
	double focusObjectDistance = this->getConfigFileField<double>("FOD");
	double focusDetectorDistance = this->getConfigFileField<double>("FDD");
	double numStartAngle = this->getConfigFileField<double>("numStartAngle");
	double denStartAngle = this->getConfigFileField<double>("denStartAngle");
	double numStopAngle = this->getConfigFileField<double>("numStopAngle");
	double denStopAngle = this->getConfigFileField<double>("denStopAngle");
	unsigned long int projectionNb = this->getConfigFileField<int>("projectionNb");
	double stepAngle = M_PI*(double)(numStopAngle/denStopAngle - numStartAngle/denStartAngle)/projectionNb;
	UniformAcquisition* acquisition = new UniformAcquisition(focusDetectorDistance,focusObjectDistance,numStartAngle/denStartAngle,numStopAngle/denStopAngle,stepAngle, projectionNb);

	return *acquisition;
}

/*NonUniformAcquisition & ConfigCT::createNonUniformAcquisition()
{
	double focusObjectDistance = this->getConfigFileField<double>("FOD");
	double focusDetectorDistance = this->getConfigFileField<double>("FDD");
	double numStartAngle = this->getConfigFileField<double>("numStartAngle");
	double denStartAngle = this->getConfigFileField<double>("denStartAngle");
	double numStopAngle = this->getConfigFileField<double>("numStopAngle");
	double denStopAngle = this->getConfigFileField<double>("denStopAngle");
	unsigned long int projectionNb = this->getConfigFileField<int>("projectionNb");

	NonUniformAcquisition* acquisition = new NonUniformAcquisition(focusDetectorDistance,focusObjectDistance,numStartAngle/denStartAngle,numStopAngle/denStopAngle,projectionNb,this->getConfigTiffFile());
	return *acquisition;
}*/

Detector & ConfigCT::createDetector()
{
	float horizontalDetectorSize = this->getConfigFileField<float>("horizontalDetectorSize");
	float verticalDetectorSize = this->getConfigFileField<float>("verticalDetectorSize");
	unsigned long int horizontalDetectorPixelNb = this->getConfigFileField<int>("horizontalDetectorPixelNb");
	unsigned long int verticalDetectorPixelNb = this->getConfigFileField<int>("verticalDetectorPixelNb");
	Detector* detector = new Detector(horizontalDetectorSize,verticalDetectorSize,horizontalDetectorPixelNb,verticalDetectorPixelNb);

	return *detector;
}

CylindricFOV & ConfigCT::createCylindricFOV()
{
	float horizontalDetectorSize = this->getConfigFileField<float>("horizontalDetectorSize");
	float verticalDetectorSize = this->getConfigFileField<float>("verticalDetectorSize");
	float focusObjectDistance = this->getConfigFileField<float>("FOD");
	float focusDetectorDistance = this->getConfigFileField<float>("FDD");
	float xFOVPixelNb = this->getConfigFileField<float>("xVolumePixelNb");
	float yFOVPixelNb = this->getConfigFileField<float>("yVolumePixelNb");
	float zFOVPixelNb = this->getConfigFileField<float>("zVolumePixelNb");
	float xMaxVolumeSize = horizontalDetectorSize*focusObjectDistance/focusDetectorDistance;
	float zMaxVolumeSize = verticalDetectorSize*(focusObjectDistance+xMaxVolumeSize/2.0)/focusDetectorDistance;
	CylindricFOV* cylindricFOV = new CylindricFOV(xMaxVolumeSize/2.0,zMaxVolumeSize, xFOVPixelNb, yFOVPixelNb, zFOVPixelNb);

	return *cylindricFOV;
}
