/*
 * Detector.cpp
 *
 *      Author: gac
 */

#include "Detector.hpp"

Detector::Detector(float uDetectorSize, float vDetectorSize, unsigned long int uDetectorPixelNb, unsigned long int vDetectorPixelNb) : uDetectorSize(uDetectorSize), vDetectorSize(vDetectorSize), uDetectorPixelNb(uDetectorPixelNb), vDetectorPixelNb(vDetectorPixelNb),uDetectorDecentering(0), vDetectorDecentering(0), detectorTilte(0)
{
	uDetectorPixelSize =  uDetectorSize/uDetectorPixelNb;
	vDetectorPixelSize = vDetectorSize/vDetectorPixelNb;
	uDetectorCenterPixel =  uDetectorPixelNb/2.0 - 0.5 + uDetectorDecentering;
	vDetectorCenterPixel = vDetectorPixelNb/2.0 - 0.5 + vDetectorDecentering;
}

Detector::Detector()
{

}
Detector::~Detector(){}

Detector & Detector::operator=(const Detector &detector)
{
	this->uDetectorSize = detector.uDetectorSize;
	this->vDetectorSize = detector.vDetectorSize;
	this->uDetectorPixelNb = detector.uDetectorPixelNb;
	this->vDetectorPixelNb = detector.vDetectorPixelNb;
	this->uDetectorDecentering = detector.uDetectorDecentering;
	this->vDetectorDecentering = detector.vDetectorDecentering;
	this->uDetectorCenterPixel = detector.uDetectorCenterPixel;
	this->vDetectorCenterPixel = detector.vDetectorCenterPixel;

	return *this;
}

float Detector::getUDetectorSize() const
{
	return uDetectorSize;
}

float Detector::getVDetectorSize() const
{
	return vDetectorSize;
}

void Detector::setUDetectorSize(float uDetectorSize)
{
	this->uDetectorSize = uDetectorSize;
}

void Detector::setVDetectorSize(float vDetectorSize)
{
	this->vDetectorSize = vDetectorSize;
}

float Detector::getUDetectorDecentering() const
{
	return uDetectorDecentering;
}

float Detector::getVDetectorDecentering() const
{
	return vDetectorDecentering;
}

void Detector::setUDetectorDecentering(float uDetectorDecentering)
{
	this->uDetectorDecentering = uDetectorDecentering;
}

void Detector::setVDetectorDecentering(float vDetectorDecentering)
{
	this->vDetectorDecentering = vDetectorDecentering;
}

float Detector::getDetectorTilte() const
{
	return detectorTilte;
}

void Detector::setDetectorTilte(float detectorTilte)
{
	this->detectorTilte = detectorTilte;
}

unsigned long int Detector::getUDetectorPixelNb() const
{
	return uDetectorPixelNb;
}

unsigned long int Detector::getVDetectorPixelNb() const
{
	return vDetectorPixelNb;
}

void Detector::setUDetectorPixelNb(unsigned long int uDetectorPixelNb)
{
	this->uDetectorPixelNb = uDetectorPixelNb;
}

void Detector::setVDetectorPixelNb(unsigned long int vDetectorPixelNb)
{
	this->vDetectorPixelNb = vDetectorPixelNb;
}

float Detector::getUDetectorPixelSize() const
{
	return uDetectorPixelSize;
}

float Detector::getVDetectorPixelSize() const
{
	return vDetectorPixelSize;
}

void Detector::setUDetectorPixelSize(float uDetectorPixelSize)
{
	this->uDetectorPixelSize = uDetectorPixelSize;
}

void Detector::setVDetectorPixelSize(float vDetectorPixelSize)
{
	this->vDetectorPixelSize = vDetectorPixelSize;
}

float Detector::getUDetectorCenterPixel() const
{
	return uDetectorCenterPixel;
}

float Detector::getVDetectorCenterPixel() const
{
	return vDetectorCenterPixel;
}

void Detector::setUDetectorCenterPixel(float uDetectorCenterPixel)
{
	this->uDetectorCenterPixel = uDetectorCenterPixel;
}

void Detector::setVDetectorCenterPixel(float vDetectorCenterPixel)
{
	this->vDetectorCenterPixel = vDetectorCenterPixel;
}



