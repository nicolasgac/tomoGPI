/*
 * Config.hpp
 *
 *      Author: gac
 */

#ifndef CONFIG_HPP_
#define CONFIG_HPP_
#ifdef __linux__
#include <unistd.h>
#endif 
#include <sys/stat.h>
#include <limits.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#ifdef __linux__ 
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#include <direct.h>
#endif

#include "Volume.cuh"
#include "Sinogram3D.cuh"
#include "Image3D.cuh"
#include "Detector.hpp"
#include "FieldOfView.hpp"
#include "ComputingArchitecture.cuh"

using namespace std;
using namespace std::chrono;

class Config{

public:

	Config();
	Config(string configDirectoryName, string configFileName);
	~Config();

	string getConfigFileName() const; // Get name of configuration file
	string getConfigDirectoryName() const; // Get name of configuration file
	void setConfigFileName(string configFileName); // Set name of configuration file
	void setConfigDirectoryName(string configDirectoryName); // Set name of configuration file
	void copyConfigFile(string destinationDirectory); // Copy configuration file in destinationDirectory
	template <typename T> T getConfigFileField(string field);

private:
	string configFileName; // Name of configuration file
	string configDirectoryName; // Name of configuration directory

};

#endif /* CONFIG_HPP_ */
