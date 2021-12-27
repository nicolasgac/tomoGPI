/*
 * ConfigCT.hpp
 *
 *      Author: gac
 */

#ifndef CONFIGCT_HPP_
#define CONFIGCT_HPP_

#include "Config.hpp"
#include "Acquisition.hpp"

class ConfigCT : public Config{

public:

	ConfigCT();
	ConfigCT(string configDirectoryName, string configFileName);
	~ConfigCT();

	UniformAcquisition& createUniformAcquisition();
	//NonUniformAcquisition& createNonUniformAcquisition();
	Detector & createDetector();
	CylindricFOV & createCylindricFOV();

	string getAcquisitionScheme();
	string getInitVolumeName();
	string getRealVolumeName();
	int getBuildRealSinogramFromTiff();


/*			Volume_CPU<T> & createVolumeCPU();
			Volume_CPU<T> & createVolumeCPU(T* dataImage);
		Volume_CPU_half  createVolumeCPU_half();
		Volume_GPU<T>  createVolumeGPU(CUDAArchitecture* cudaArchitecture);
		Volume_GPU_half  createVolumeGPU_half(CUDAArchitecture* cudaArchitecture);*/



private:
	string acquisitionScheme;
	string initVolumeName;
	string realVolumeName;
	int buildRealSinogramFromTiff;
};


#endif /* CONFIGCT_HPP_ */
