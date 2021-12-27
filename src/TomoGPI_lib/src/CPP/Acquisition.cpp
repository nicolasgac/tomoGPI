/*
 /* Acquisition.cpp
 *
 *      Author: gac
 */


#include "Acquisition.hpp"

Acquisition::Acquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, unsigned long int projectionNb) : focusDetectorDistance(focusDetectorDistance), focusObjectDistance(focusObjectDistance), startAngle(startAngle), stopAngle(stopAngle), projectionNb(projectionNb), phiValue(0){}

Acquisition::~Acquisition()
{
	delete phiValue;
}

Acquisition & Acquisition::operator=(const Acquisition &acquisition)
{
	this->focusDetectorDistance = acquisition.focusDetectorDistance;
	this->focusObjectDistance = acquisition.focusObjectDistance;
	this->startAngle = acquisition.startAngle;
	this->stopAngle = acquisition.stopAngle;
	//this->stepAngle = acquisition.stepAngle;
	this->projectionNb = acquisition.projectionNb;
	this->phiValue = acquisition.phiValue;
	return *this;
}

double Acquisition::getFocusDetectorDistance() const
{
	return focusDetectorDistance;
}

double Acquisition::getFocusObjectDistance() const
{
	return focusObjectDistance;
}

void Acquisition::setFocusDetectorDistance(double focusDetectorDistance)
{
	this->focusDetectorDistance = focusDetectorDistance;
}

void Acquisition::setFocusObjectDistance(double focusObjectDistance)
{
	this->focusObjectDistance = focusObjectDistance;
}

double Acquisition::getStartAngle() const
{
	return startAngle;
}

double Acquisition::getStopAngle() const
{
	return stopAngle;
}

unsigned long int Acquisition::getProjectionNb() const
{
	return projectionNb;
}

double* Acquisition::getPhiValue()
{
	return phiValue;
}

void Acquisition::setStartAngle(double startAngle)
{
	this->startAngle = startAngle;
}

void Acquisition::setStopAngle(double stopAngle)
{
	this->stopAngle = stopAngle;
}

void Acquisition::setProjectionNb(unsigned long int projectionNb)
{
	this->projectionNb = projectionNb;
}

void Acquisition::setPhiValue(double* phiValue)
{
	this->phiValue = phiValue;
}

// UniformAcquisition definition
UniformAcquisition::UniformAcquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, double stepAngle, unsigned long int projectionNb) : Acquisition(focusDetectorDistance, focusObjectDistance, startAngle, stopAngle, projectionNb), stepAngle(stepAngle)
{
	this->setPhiValue(new double[projectionNb]);
	for (int p=0;p<projectionNb;p++)
	{
		this->getPhiValue()[p] = (p+startAngle)*stepAngle;
	}
}

UniformAcquisition::~UniformAcquisition(){}

UniformAcquisition & UniformAcquisition::operator=(const UniformAcquisition &acquisition)
{
	this->stepAngle = acquisition.stepAngle;
	return *this;
}

double UniformAcquisition::getStepAngle() const
{
	return stepAngle;
}

void UniformAcquisition::setStepAngle(double stepAngle)
{
	this->stepAngle = stepAngle;
}

// NonUniformAcquisition definition
template<typename T>
NonUniformAcquisition<T>::NonUniformAcquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, unsigned long int projectionNb, ConfigTiff<T>* configTiffFile) : Acquisition(focusDetectorDistance, focusObjectDistance, startAngle, stopAngle, projectionNb)
{
	projectionNb = configTiffFile->getUsedProjectionNb();
	this->setPhiValue(new double[projectionNb]);
	for (int p=0;p<projectionNb;p++)
	{
		this->getPhiValue()[p] = configTiffFile->getPhiValue()[p];
	}
}

template <typename T>
NonUniformAcquisition<T>::~NonUniformAcquisition(){}

template class NonUniformAcquisition<unsigned char>; // 8-bit unsigned sinogram
template class NonUniformAcquisition<unsigned short>; // 16-bit unsigned sinogram




