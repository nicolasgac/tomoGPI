/*
 * FieldOfView.cpp
 *
 *      Author: gac
 */

#include "FieldOfView.hpp"

FieldOfView::FieldOfView(){}
FieldOfView::~FieldOfView(){}

/* Cylindric FOV definition */
CylindricFOV::CylindricFOV(float cylinderRadius, float cylinderHeight, float xFOVPixelNb, float yFOVPixelNb, float zFOVPixelNb) : cylinderRadius(cylinderRadius), cylinderHeight(cylinderHeight), xFOVPixelNb(xFOVPixelNb), yFOVPixelNb(yFOVPixelNb), zFOVPixelNb(zFOVPixelNb){}

CylindricFOV::~CylindricFOV(){}

CylindricFOV & CylindricFOV::operator=(const CylindricFOV &cylindricFOV)
{
	this->cylinderRadius = cylindricFOV.cylinderRadius;
	this->cylinderHeight = cylindricFOV.cylinderHeight;

	return *this;
}

float CylindricFOV::getXFOVSize()
{
	return 2.0*cylinderRadius;
}

float CylindricFOV::getYFOVSize()
{
	return 2.0*cylinderRadius;
}

float CylindricFOV::getZFOVSize()
{
	return cylinderHeight;
}

float CylindricFOV::getXFOVPixelNb()
{
	return this->xFOVPixelNb;
}

float CylindricFOV::getYFOVPixelNb()
{
	return this->yFOVPixelNb;
}

float CylindricFOV::getZFOVPixelNb()
{
	return this->zFOVPixelNb;
}

void CylindricFOV::setXFOVSize(float xFOVSize)
{
	this->cylinderRadius = xFOVSize/2.0;
}

void CylindricFOV::setYFOVSize(float yFOVSize)
{
	this->cylinderRadius = yFOVSize/2.0;
}

void CylindricFOV::setZFOVSize(float zFOVSize)
{
	this->cylinderHeight = zFOVSize;
}

void CylindricFOV::setXFOVPixelNb(float xFOVPixelNb)
{
	this->xFOVPixelNb = xFOVPixelNb;
}

void CylindricFOV::setYFOVPixelNb(float yFOVPixelNb)
{
	this->yFOVPixelNb = yFOVPixelNb;
}

void CylindricFOV::setZFOVPixelNb(float zFOVPixelNb)
{
	this->zFOVPixelNb = zFOVPixelNb;
}


