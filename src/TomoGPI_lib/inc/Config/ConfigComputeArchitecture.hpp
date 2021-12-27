/*
 * ConfigComputeArchitecture.hpp
 *
 *      Author: gac
 */

#ifndef CONFIGGPU_HPP_
#define CONFIGGPU_HPP_

#include "Config.hpp"
#include "Acquisition.hpp"




class ConfigComputeArchitecture : public Config{

public:

	ConfigComputeArchitecture();
	ConfigComputeArchitecture(string configDirectoryName, string configFileName);
	~ConfigComputeArchitecture();

	CUDAProjectionArchitecture & createCUDAProjectionArchitecture();
	CUDABProjectionArchitecture & createCUDABProjectionArchitecture(FieldOfView* fov);
	CUDAArchitecture & createCUDAArchitectureSino(Detector* detector, Acquisition* acquisition);
	CUDAArchitecture & createCUDAArchitectureVolume(FieldOfView* fov);

	OCLProjectionArchitecture & createOCLProjectionArchitecture();
	OCLBProjectionArchitecture & createOCLBProjectionArchitecture(FieldOfView* fov);
	OCLArchitecture & createOCLArchitectureSino(Detector* detector, Acquisition* acquisition);
	OCLArchitecture & createOCLArchitectureVolume(FieldOfView* fov);

	kind_precision getPrecision();
	kind_storage getStorage();
	kind_pair getPair();
	kind_compute getCompute();
	kind_architecture getArchitecture() const; // 

	unsigned short getGpuNb_proj();
	unsigned short getGpuNb_back();
	unsigned short getGpuNb_sino();
	unsigned short getGpuNb_vol();
unsigned short getprojectionStreamsNb();
unsigned short getbackprojectionStreamsNb();


	void setArchitecture(kind_architecture architecture); // 

	void setGpuNb_proj(unsigned short n);
	void setGpuNb_back(unsigned short n);
	void setGpuNb_sino(unsigned short n);
	void setGpuNb_vol(unsigned short n);
	void setProjectionStreamsNb(unsigned short n);
void setBProjectionStreamsNb(unsigned short n);
void setPrecision(kind_precision precision);
	void setStorage(kind_storage storage );
void setPair(kind_pair pair);
	void setCompute(kind_compute compute);

private:
	kind_precision precision;
	kind_storage storage;
	kind_compute compute;
	kind_pair pair;
	kind_architecture architecture;
unsigned short gpuNb_proj;
unsigned short gpuNb_back;
unsigned short gpuNb_sino;
unsigned short gpuNb_vol;
unsigned short projectionStreamsNb;
unsigned short backprojectionStreamsNb;


};


#endif /* CONFIGGPU_HPP_ */
