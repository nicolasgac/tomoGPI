/*
 * ConfigComputeArchitecture.cpp
 *
  *      Author: gac
 */

#include "ConfigComputeArchitecture.hpp"

ConfigComputeArchitecture::ConfigComputeArchitecture() : Config() , precision(FLOAT_GPU), storage(MEM_CPU), compute(COMPUTE_CUDA), pair(PAIR_RSVI){}
ConfigComputeArchitecture::ConfigComputeArchitecture(string configDirectoryName, string configFileName) : Config(configDirectoryName, configFileName)
{
	if (strcmp(this->getConfigFileField<string>("precision").c_str(),"half")==0)
		this->precision = HALF_GPU;
	else if (strcmp(this->getConfigFileField<string>("precision").c_str(),"double")==0)
		this->precision = DOUBLE_GPU;
	else
		this->precision = FLOAT_GPU;

	if (strcmp(this->getConfigFileField<string>("storage").c_str(),"mem_GPU")==0)
		this->storage = MEM_GPU;
	else if (strcmp(this->getConfigFileField<string>("storage").c_str(),"mem_MGPU")==0)
		this->storage = MEM_MGPU;
	else
		this->storage=MEM_CPU;

if (strcmp(this->getConfigFileField<string>("compute").c_str(),"OCL")==0)
		this->compute = COMPUTE_OCL;
	else if (strcmp(this->getConfigFileField<string>("compute").c_str(),"C")==0)
		this->compute = COMPUTE_C;
	else
		this->compute= COMPUTE_CUDA;



		if (strcmp(this->getConfigFileField<string>("architecture").c_str(),"CPU")==0)
		this->architecture = ARCHITECTURE_CPU;
	else if (strcmp(this->getConfigFileField<string>("architecture").c_str(),"FPGA")==0)
		this->architecture = ARCHITECTURE_FPGA;
	else
		this->architecture= ARCHITECTURE_GPU;

 if (strcmp(this->getConfigFileField<string>("pair").c_str(),"SiddonVI")==0)
		this->pair = PAIR_SIDDONVI;
	else
		this->pair= PAIR_RSVI;


}
ConfigComputeArchitecture::~ConfigComputeArchitecture(){}

kind_precision ConfigComputeArchitecture::getPrecision()
{
	return this->precision;
}



kind_storage ConfigComputeArchitecture::getStorage()
{
	return this->storage;
}


kind_pair ConfigComputeArchitecture::getPair()
{
	return this->pair;
}

kind_compute ConfigComputeArchitecture::getCompute()
{
	return this->compute;
}


kind_architecture ConfigComputeArchitecture::getArchitecture() const
{
	return this->architecture;
}

void ConfigComputeArchitecture::setArchitecture(kind_architecture architecture)
{
	this->architecture = architecture;
}


CUDAProjectionArchitecture & ConfigComputeArchitecture::createCUDAProjectionArchitecture()
{
	unsigned short xBlockNb = -1;//this->getConfigFileField<int>("xBlockNbP");
	unsigned short yBlockNb = -1;//this->getConfigFileField<int>("yBlockNbP");
	unsigned short xThreadNb = this->getConfigFileField<int>("xThreadNbP");
	unsigned short yThreadNb = this->getConfigFileField<int>("yThreadNbP");
	unsigned short numberOfCells = this->getConfigFileField<int>("numberOfCellsP");
	unsigned short projectionThreadNb = this->getConfigFileField<int>("projectionThreadNbP");
	unsigned short projectionStreamsNb = this->getConfigFileField<int>("streamsNbP");
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbP");

	CUDAProjectionArchitecture* cudaProjectionArchitecture = new CUDAProjectionArchitecture(gpuNb, xBlockNb, yBlockNb, xThreadNb, yThreadNb, projectionThreadNb, projectionStreamsNb, numberOfCells);

	return *cudaProjectionArchitecture;
}

CUDABProjectionArchitecture & ConfigComputeArchitecture::createCUDABProjectionArchitecture(FieldOfView* fov)
{
	unsigned short xThreadNb = this->getConfigFileField<int>("xThreadNbBP");
	unsigned short yThreadNb = this->getConfigFileField<int>("yThreadNbBP");
	unsigned short numberOfVoxels = this->getConfigFileField<int>("numberOfVoxelsBP");
	unsigned short xBlockNb = fov->getXFOVPixelNb()/xThreadNb;
	unsigned short yBlockNb = fov->getYFOVPixelNb()/yThreadNb;
	unsigned short projectionLoopSize = this->getConfigFileField<int>("projectionLoopSizeBP");
	unsigned short bprojectionStreamsNb = this->getConfigFileField<int>("streamsNbBP");
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbBP");

	CUDABProjectionArchitecture* cudaBProjectionArchitecture = new CUDABProjectionArchitecture(gpuNb, xBlockNb, yBlockNb, xThreadNb, yThreadNb, projectionLoopSize, bprojectionStreamsNb, numberOfVoxels);

	return *cudaBProjectionArchitecture;
}

CUDAArchitecture & ConfigComputeArchitecture::createCUDAArchitectureSino(Detector* detector, Acquisition* acquisition)
{
	unsigned short uThreadNb = this->getConfigFileField<int>("uThreadNbSino");
	unsigned short vThreadNb = this->getConfigFileField<int>("vThreadNbSino");
	unsigned short pThreadNb = this->getConfigFileField<int>("pThreadNbSino");
	unsigned short uBlockNb = detector->getUDetectorPixelNb()/uThreadNb;
	unsigned short vBlockNb = detector->getVDetectorPixelNb()/vThreadNb;
	unsigned short pBlockNb = acquisition->getProjectionNb()/pThreadNb;
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbSino");

	CUDAArchitecture* cudaArchitecture = new CUDAArchitecture(gpuNb,uBlockNb,vBlockNb,pBlockNb,uThreadNb,vThreadNb,pThreadNb);

	return *cudaArchitecture;
}

CUDAArchitecture & ConfigComputeArchitecture::createCUDAArchitectureVolume(FieldOfView* fov)
{
	unsigned short xThreadNbVolume = this->getConfigFileField<int>("xThreadNbVolume");
	unsigned short yThreadNbVolume = this->getConfigFileField<int>("yThreadNbVolume");
	unsigned short zThreadNbVolume = this->getConfigFileField<int>("zThreadNbVolume");
	unsigned short xBlockNbVolume = fov->getXFOVPixelNb()/xThreadNbVolume;
	unsigned short yBlockNbVolume = fov->getXFOVPixelNb()/yThreadNbVolume;
	unsigned short zBlockNbVolume = fov->getXFOVPixelNb()/zThreadNbVolume;
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbVolume");

	CUDAArchitecture* cudaArchitecture = new CUDAArchitecture(gpuNb,xBlockNbVolume,yBlockNbVolume,zBlockNbVolume,xThreadNbVolume,yThreadNbVolume,zThreadNbVolume);

	return *cudaArchitecture;
}

OCLProjectionArchitecture & ConfigComputeArchitecture::createOCLProjectionArchitecture()
{
	unsigned short xBlockNb = -1;//this->getConfigFileField<int>("xBlockNbP");
	unsigned short yBlockNb = -1;//this->getConfigFileField<int>("yBlockNbP");
	unsigned short xThreadNb = this->getConfigFileField<int>("xThreadNbP");
	unsigned short yThreadNb = this->getConfigFileField<int>("yThreadNbP");
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbP");

	OCLProjectionArchitecture* oclProjectionArchitecture = new OCLProjectionArchitecture(gpuNb, xBlockNb, yBlockNb, xThreadNb, yThreadNb);

	return *oclProjectionArchitecture;
}

OCLBProjectionArchitecture & ConfigComputeArchitecture::createOCLBProjectionArchitecture(FieldOfView* fov)
{
	unsigned short xThreadNb = this->getConfigFileField<int>("xThreadNbBP");
	unsigned short yThreadNb = this->getConfigFileField<int>("yThreadNbBP");
	unsigned short xBlockNb = fov->getXFOVPixelNb()/xThreadNb;
	unsigned short yBlockNb = fov->getYFOVPixelNb()/yThreadNb;
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbBP");

	OCLBProjectionArchitecture* oclBProjectionArchitecture = new OCLBProjectionArchitecture(gpuNb, xBlockNb, yBlockNb, xThreadNb, yThreadNb);

	return *oclBProjectionArchitecture;
}

OCLArchitecture & ConfigComputeArchitecture::createOCLArchitectureSino(Detector* detector, Acquisition* acquisition)
{
	unsigned short uThreadNb = this->getConfigFileField<int>("uThreadNbSino");
	unsigned short vThreadNb = this->getConfigFileField<int>("vThreadNbSino");
	unsigned short pThreadNb = this->getConfigFileField<int>("pThreadNbSino");
	unsigned short uBlockNb = detector->getUDetectorPixelNb()/uThreadNb;
	unsigned short vBlockNb = detector->getVDetectorPixelNb()/vThreadNb;
	unsigned short pBlockNb = acquisition->getProjectionNb()/pThreadNb;
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbSino");

	OCLArchitecture* oclArchitecture = new OCLArchitecture(gpuNb,uBlockNb,vBlockNb,pBlockNb,uThreadNb,vThreadNb,pThreadNb);

	return *oclArchitecture;
}

OCLArchitecture & ConfigComputeArchitecture::createOCLArchitectureVolume(FieldOfView* fov)
{
	unsigned short xThreadNbVolume = this->getConfigFileField<int>("xThreadNbVolume");
	unsigned short yThreadNbVolume = this->getConfigFileField<int>("yThreadNbVolume");
	unsigned short zThreadNbVolume = this->getConfigFileField<int>("zThreadNbVolume");
	unsigned short xBlockNbVolume = fov->getXFOVPixelNb()/xThreadNbVolume;
	unsigned short yBlockNbVolume = fov->getXFOVPixelNb()/yThreadNbVolume;
	unsigned short zBlockNbVolume = fov->getXFOVPixelNb()/zThreadNbVolume;
	unsigned short gpuNb = this->getConfigFileField<int>("gpuNbVolume");

	OCLArchitecture* oclArchitecture = new OCLArchitecture(gpuNb,xBlockNbVolume,yBlockNbVolume,zBlockNbVolume,xThreadNbVolume,yThreadNbVolume,zThreadNbVolume);

	return *oclArchitecture;
}

unsigned short ConfigComputeArchitecture::getGpuNb_proj(){
return this->gpuNb_proj;
}
	unsigned short ConfigComputeArchitecture::getGpuNb_back(){
return this->gpuNb_back;
}
	unsigned short ConfigComputeArchitecture::getGpuNb_sino(){
return this->gpuNb_sino;
}
	unsigned short ConfigComputeArchitecture::getGpuNb_vol(){
return this->gpuNb_vol;
}

	unsigned short ConfigComputeArchitecture::getprojectionStreamsNb(){
return this->projectionStreamsNb;
	}

unsigned short ConfigComputeArchitecture::getbackprojectionStreamsNb(){
return this->backprojectionStreamsNb;
}

void ConfigComputeArchitecture::setPrecision(kind_precision precision){
this->precision=precision;
}
	void ConfigComputeArchitecture::setStorage(kind_storage storage ){
this->storage=storage;
	}

void ConfigComputeArchitecture::setPair(kind_pair pair){
this->pair=pair;

}
	void ConfigComputeArchitecture::setCompute(kind_compute compute ){
this->compute=compute;
	}

	void ConfigComputeArchitecture::setGpuNb_proj(unsigned short n){
this->gpuNb_proj=n;
	}

	void ConfigComputeArchitecture::setGpuNb_back(unsigned short n){
this->gpuNb_back=n;
	}

	void ConfigComputeArchitecture::setGpuNb_sino(unsigned short n){
this->gpuNb_sino=n;
	}

	void ConfigComputeArchitecture::setGpuNb_vol(unsigned short n){
this->gpuNb_vol=n;
	}

	void ConfigComputeArchitecture::setProjectionStreamsNb(unsigned short n){
this->projectionStreamsNb=n;
	}

void ConfigComputeArchitecture::setBProjectionStreamsNb(unsigned short n){
this->backprojectionStreamsNb=n;
}
