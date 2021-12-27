/*
 * ConfigTiff.cpp
 *
 *      Author: gac
 */

#include "ConfigTiff.hpp"

template<typename V>
ConfigTiff<V>::ConfigTiff() : Config(), tiffName(string()), totalProjectionNb(0), usedProjectionNb(0), uDetectorPixelInitialNb(0), vDetectorPixelInitialNb(0), uDetectorPixelFinalNb(0), vDetectorPixelFinalNb(0), projectionSamplingRate(0), phiValue(NULL), normalizationFactor(0){}

template<typename V>
ConfigTiff<V>::ConfigTiff(string configDirectoryName, string configFileName) : Config(configDirectoryName, configFileName)
{
	tiffName = getConfigFileField<string>("tiffName");
	totalProjectionNb = getConfigFileField<int>("totalProjectionNb");
	usedProjectionNb = getConfigFileField<int>("usedProjectionNb");
	uDetectorPixelInitialNb = getConfigFileField<int>("uDetectorPixelInitialNb");
	vDetectorPixelInitialNb = getConfigFileField<int>("vDetectorPixelInitialNb");
	uDetectorPixelFinalNb = getConfigFileField<int>("uDetectorPixelFinalNb");
	vDetectorPixelFinalNb = getConfigFileField<int>("vDetectorPixelFinalNb");
	uDetectorCenterFinal = getConfigFileField<int>("uDetectorCenterFinal");
	vDetectorCenterFinal = getConfigFileField<int>("vDetectorCenterFinal");
	projectionSamplingRate = totalProjectionNb/usedProjectionNb;
	normalizationFactor = getConfigFileField<double>("normalizationFactor");
	phiValue = NULL;
}

template<typename V>
ConfigTiff<V>::~ConfigTiff(){};

template<typename V>
string ConfigTiff<V>::getTiffName()
{
	return tiffName;
}

template<typename V>
unsigned short ConfigTiff<V>::getTotalProjectionNb()
{
	return totalProjectionNb;
}

template<typename V>
unsigned short ConfigTiff<V>::getUsedProjectionNb()
{
	return usedProjectionNb;
}

template<typename V>
unsigned short ConfigTiff<V>::getUDetectorPixelInitialNb()
{
	return uDetectorPixelInitialNb;
}

template<typename V>
unsigned short ConfigTiff<V>::getVDetectorPixelInitialNb()
{
	return vDetectorPixelInitialNb;
}

template<typename V>
unsigned short ConfigTiff<V>::getUDetectorPixelFinalNb()
{
	return uDetectorPixelFinalNb;
}

template<typename V>
unsigned short ConfigTiff<V>::getVDetectorPixelFinalNb()
{
	return vDetectorPixelFinalNb;
}


template<typename V>
unsigned short ConfigTiff<V>::getUDetectorCenterFinal()
{
	return uDetectorCenterFinal;
}


template<typename V>
unsigned short ConfigTiff<V>::getVDetectorCenterFinal()
{
	return vDetectorCenterFinal;
}


template<typename V>
unsigned short ConfigTiff<V>::getProjectionSamplingRate()
{
	return projectionSamplingRate;
}
template<typename V>
double* ConfigTiff<V>::getPhiValue()
{
	return phiValue;
}

template<typename V>
void ConfigTiff<V>::setTiffName(string tiffName)
{
	this->tiffName = tiffName;
}

template<typename V>
void ConfigTiff<V>::setTotalProjectionNb(unsigned short totalProjectionNb)
{
	this->totalProjectionNb = totalProjectionNb;
}

template<typename V>
void ConfigTiff<V>::setUsedProjectionNb(unsigned short usedProjectionNb)
{
	this->usedProjectionNb = usedProjectionNb;
}

template<typename V>
void ConfigTiff<V>::setUDetectorPixelInitialNb(unsigned short uDetectorPixelInitialNb)
{
	this->uDetectorPixelInitialNb = uDetectorPixelInitialNb;
}

template<typename V>
void ConfigTiff<V>::setVDetectorPixelInitialNb(unsigned short vDetectorPixelInitialNb)
{
	this->vDetectorPixelInitialNb = vDetectorPixelInitialNb;
}

template<typename V>
void ConfigTiff<V>::setUDetectorPixelFinalNb(unsigned short uDetectorPixelFinalNb)
{
	this->uDetectorPixelFinalNb = uDetectorPixelFinalNb;
}

template<typename V>
void ConfigTiff<V>::setVDetectorPixelFinalNb(unsigned short vDetectorPixelFinalNb)
{
	this->vDetectorPixelFinalNb = vDetectorPixelFinalNb;
}


template<typename V>
void ConfigTiff<V>::setUDetectorCenterFinal(unsigned short uDetectorCenterFinal)
{
	this->uDetectorCenterFinal = uDetectorCenterFinal;
}

template<typename V>
void ConfigTiff<V>::setVDetectorCenterFinal(unsigned short vDetectorCenterFinal)
{
	this->vDetectorCenterFinal = vDetectorCenterFinal;
}

template<typename V>
void ConfigTiff<V>::setProjectionSamplingRate(unsigned short projectionSamplingRate)
{
	this->projectionSamplingRate = projectionSamplingRate;
}



template class ConfigTiff<unsigned char>; // 8-bit unsigned sinogram
template class ConfigTiff<unsigned short>; // 16-bit unsigned sinogram


