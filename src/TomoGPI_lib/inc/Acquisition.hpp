/*
 * Acquisition.hpp
 *
 *      Author: gac
 */

#ifndef ACQUISITION_HPP_
#define ACQUISITION_HPP_

#ifdef __linux__ 
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "ConfigTiff.hpp"
#include "Config.hpp"

#define FOV_CYLINDER 0

class Acquisition{

public:

	Acquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, unsigned long int projectionNb); // Constructor
	~Acquisition(); // Destructor

	Acquisition & operator=(const Acquisition &acquisition);

	/* Distance Parameters*/
	double getFocusDetectorDistance() const; // Get focus detector distance
	double getFocusObjectDistance() const; // Get focus object distance
	void setFocusDetectorDistance(double focusDetectorDistance); // Set focus detector distance
	void setFocusObjectDistance(double focusObjectDistance); // Set focus object distance

	/* Angle Parameters*/
	double getStartAngle() const; // Get Start angle (radian)
	double getStopAngle() const; // Get Stop angle (radian)
	unsigned long int getProjectionNb() const; // Get projection number during acquisition
	double* getPhiValue(); // Get Value of angle of projections
	void setStartAngle(double startAngle); // Get Start angle (radian)
	void setStopAngle(double stopAngle); // Set Stop angle (radian)
	void setProjectionNb(unsigned long int projectionNb); // Set projection number during acquisition
	void setPhiValue(double* phiValue); // Set Value of angle of projections

private:
	/* Distance Parameters*/
	double focusDetectorDistance; // Focus detector distance
	double focusObjectDistance; // Focus object distance

	/* Angle Parameters*/
	double startAngle; // Start angle (radian)
	double stopAngle; // Stop angle (radian)
	unsigned long int projectionNb; // Projection number during acquisition
	double* phiValue; // Value of angle of projections
};

class UniformAcquisition : public Acquisition{

public:
	UniformAcquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, double stepAngle, unsigned long int projectionNb); // Constructor
	~UniformAcquisition();

	UniformAcquisition & operator=(const UniformAcquisition &acquisition);

	double getStepAngle() const; // Get Angle step (radian)
	void setStepAngle(double stepAngle); // Set Angle step (radian)

private:
	double stepAngle; // Angle step (radian)

};

template<typename T> class NonUniformAcquisition : public Acquisition{

public:
	NonUniformAcquisition(double focusDetectorDistance, double focusObjectDistance, double startAngle, double stopAngle, unsigned long int projectionNb, ConfigTiff<T>* configTiffFile); // Constructor
	~NonUniformAcquisition();
};

#endif /* ACQUISITION_HPP_ */
