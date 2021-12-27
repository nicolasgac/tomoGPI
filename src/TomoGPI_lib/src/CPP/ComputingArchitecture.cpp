/*
 * ComputingArchitecture.cpp
 *
 *      Author: gac
 */


#include "ComputingArchitecture.cuh"




ComputingArchitecture::ComputingArchitecture() : architecture(ARCHITECTURE_GPU), compute(COMPUTE_CUDA), computingUnitNb(0){}
ComputingArchitecture::ComputingArchitecture(kind_architecture architecture, kind_compute compute, unsigned short computingUnitNb) : architecture(architecture), compute(compute),computingUnitNb(computingUnitNb){}
ComputingArchitecture::~ComputingArchitecture(){}

ComputingArchitecture & ComputingArchitecture::operator=(const ComputingArchitecture &computingArchitecture)
{
	this->computingUnitNb = computingArchitecture.computingUnitNb;
	this->architecture = computingArchitecture.architecture;
	this->compute = computingArchitecture.compute;

	return *this;
}

unsigned short ComputingArchitecture::getComputingUnitNb() const
{
	return this->computingUnitNb;
}

void ComputingArchitecture::setComputingUnitNb(unsigned short computingUnitNb)
{
	this->computingUnitNb = computingUnitNb;
}


kind_architecture ComputingArchitecture::getArchitecture() const
{
	return this->architecture;
}

void ComputingArchitecture::setArchitecture(kind_architecture architecture)
{
	this->architecture = architecture;
}

kind_compute ComputingArchitecture::getCompute() const
{
	return this->compute;
}

void ComputingArchitecture::setCompute(kind_compute compute)
{
	this->compute = compute;
}


/* CUDAArchitecture definition */
CUDAArchitecture::CUDAArchitecture() : ComputingArchitecture(), xBlockNb(0), yBlockNb(0), zBlockNb(0), xThreadNb(0), yThreadNb(0), zThreadNb(0){}
CUDAArchitecture::CUDAArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short zBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short zThreadNb) :  ComputingArchitecture(ARCHITECTURE_GPU, COMPUTE_CUDA,gpuNb), xBlockNb(xBlockNb), yBlockNb(yBlockNb), zBlockNb(zBlockNb), xThreadNb(xThreadNb), yThreadNb(yThreadNb), zThreadNb(zThreadNb){}
CUDAArchitecture::~CUDAArchitecture(){}

CUDAArchitecture::CUDAArchitecture(const CUDAArchitecture & cudaArchitectureTocopy)
{
	this->xBlockNb = cudaArchitectureTocopy.xBlockNb;
	this->yBlockNb = cudaArchitectureTocopy.yBlockNb;
	this->zBlockNb = cudaArchitectureTocopy.zBlockNb;
	this->xThreadNb = cudaArchitectureTocopy.xThreadNb;
	this->yThreadNb = cudaArchitectureTocopy.yThreadNb;
	this->zThreadNb = cudaArchitectureTocopy.zThreadNb;
	this->setComputingUnitNb(cudaArchitectureTocopy.getComputingUnitNb());
}

CUDAArchitecture & CUDAArchitecture::operator=(const CUDAArchitecture & cudaArchitecture)
{
	this->xBlockNb = cudaArchitecture.xBlockNb;
	this->yBlockNb = cudaArchitecture.yBlockNb;
	this->zBlockNb = cudaArchitecture.zBlockNb;
	this->xThreadNb = cudaArchitecture.xThreadNb;
	this->yThreadNb = cudaArchitecture.yThreadNb;
	this->zThreadNb = cudaArchitecture.zThreadNb;
	this->setComputingUnitNb(cudaArchitecture.getComputingUnitNb());

	return *this;
}

unsigned short CUDAArchitecture::getXBlockNb() const
{
	return xBlockNb;
}

unsigned short CUDAArchitecture::getYBlockNb() const
{
	return yBlockNb;
}

unsigned short CUDAArchitecture::getZBlockNb() const
{
	return zBlockNb;
}

unsigned short CUDAArchitecture::getXThreadNb() const
{
	return xThreadNb;
}

unsigned short CUDAArchitecture::getYThreadNb() const
{
	return yThreadNb;
}

unsigned short CUDAArchitecture::getZThreadNb() const
{
	return zThreadNb;
}

void CUDAArchitecture::setXBlockNb(unsigned short xBlockNb)
{
	this->xBlockNb = xBlockNb;
}

void CUDAArchitecture::setYBlockNb(unsigned short yBlockNb)
{
	this->yBlockNb = yBlockNb;
}

void CUDAArchitecture::setZBlockNb(unsigned short zBlockNb)
{
	this->zBlockNb = zBlockNb;
}

void CUDAArchitecture::setXThreadNb(unsigned short xThreadNb)
{
	this->xThreadNb = xThreadNb;
}

void CUDAArchitecture::setYThreadNb(unsigned short yThreadNb)
{
	this->yThreadNb = yThreadNb;
}

void CUDAArchitecture::setZThreadNb(unsigned short zThreadNb)
{
	this->zThreadNb = zThreadNb;
}

/* CUDAProjectionArchitecture definition */
CUDAProjectionArchitecture::CUDAProjectionArchitecture() : CUDAArchitecture(), projectionThreadNb(0), projectionStreamsNb(0), numberOfCells(0){}
CUDAProjectionArchitecture::CUDAProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short projectionThreadNb, unsigned short projectionStreamsNb, unsigned short numberOfCells) : CUDAArchitecture(gpuNb, xBlockNb, yBlockNb, 1, xThreadNb, yThreadNb, 1), projectionThreadNb(projectionThreadNb), projectionStreamsNb(projectionStreamsNb), numberOfCells(numberOfCells){}
CUDAProjectionArchitecture::~CUDAProjectionArchitecture(){}

CUDAProjectionArchitecture & CUDAProjectionArchitecture::operator=(const CUDAProjectionArchitecture & cudaProjectionArchitecture)
{
	this->projectionThreadNb = cudaProjectionArchitecture.projectionThreadNb;
	this->projectionStreamsNb = cudaProjectionArchitecture.projectionStreamsNb;
	this->numberOfCells = cudaProjectionArchitecture.numberOfCells;
	this->setComputingUnitNb(cudaProjectionArchitecture.getComputingUnitNb());
	this->setXBlockNb(cudaProjectionArchitecture.getXBlockNb());
	this->setYBlockNb(cudaProjectionArchitecture.getYBlockNb());
	this->setXThreadNb(cudaProjectionArchitecture.getXThreadNb());
	this->setYThreadNb(cudaProjectionArchitecture.getYThreadNb());

	return *this;
}

unsigned short CUDAProjectionArchitecture::getProjectionThreadNb() const
{
	return projectionThreadNb;
}

unsigned short CUDAProjectionArchitecture::getProjectionStreamsNb() const
{
	return projectionStreamsNb;
}

unsigned short CUDAProjectionArchitecture::getNumberOfCells() const
{
	return numberOfCells;
}



void CUDAProjectionArchitecture::setProjectionThreadNb(unsigned short projectionThreadNb)
{
	this->projectionThreadNb = projectionThreadNb;
}

void CUDAProjectionArchitecture::setProjectionStreamsNb(unsigned short projectionStreamsNb)
{
	this->projectionStreamsNb = projectionStreamsNb;
}

void CUDAProjectionArchitecture::setNumberOfCells(unsigned short numberOfCells)
{
	this->numberOfCells = numberOfCells;
}

/* CUDABProjectionArchitecture definition */
CUDABProjectionArchitecture::CUDABProjectionArchitecture() : CUDAArchitecture(), projectionLoopSize(0), bprojectionStreamsNb(0), numberOfVoxels(0){}
CUDABProjectionArchitecture::CUDABProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short projectionLoopSize, unsigned short bprojectionStreamsNb, unsigned short numberOfVoxels) : CUDAArchitecture(gpuNb, xBlockNb, yBlockNb, 1, xThreadNb, yThreadNb, 1), projectionLoopSize(projectionLoopSize), bprojectionStreamsNb(bprojectionStreamsNb), numberOfVoxels(numberOfVoxels){}
CUDABProjectionArchitecture::~CUDABProjectionArchitecture(){}

CUDABProjectionArchitecture & CUDABProjectionArchitecture::operator=(const CUDABProjectionArchitecture & cudaBProjectionArchitecture)
{
	this->projectionLoopSize = cudaBProjectionArchitecture.projectionLoopSize;
	this->bprojectionStreamsNb = cudaBProjectionArchitecture.bprojectionStreamsNb;
	this->numberOfVoxels = cudaBProjectionArchitecture.numberOfVoxels;
	this->setComputingUnitNb(cudaBProjectionArchitecture.getComputingUnitNb());
	this->setXBlockNb(cudaBProjectionArchitecture.getXBlockNb());
	this->setYBlockNb(cudaBProjectionArchitecture.getYBlockNb());
	this->setXThreadNb(cudaBProjectionArchitecture.getXThreadNb());
	this->setYThreadNb(cudaBProjectionArchitecture.getYThreadNb());

	return *this;
}

unsigned short CUDABProjectionArchitecture::getProjectionLoopSize() const
{
	return projectionLoopSize;
}

unsigned short CUDABProjectionArchitecture::getBProjectionStreamsNb() const
{
	return bprojectionStreamsNb;
}

unsigned short CUDABProjectionArchitecture::getNumberOfVoxels() const
{
	return numberOfVoxels;
}

void CUDABProjectionArchitecture::setProjectionLoopSize(unsigned short projectionLoopSize)
{
	this->projectionLoopSize = projectionLoopSize;
}

void CUDABProjectionArchitecture::setBProjectionStreamsNb(unsigned short bprojectionStreamsNb)
{
	this->bprojectionStreamsNb = bprojectionStreamsNb;
}

void CUDABProjectionArchitecture::setNumberOfVoxels(unsigned short numberOfVoxels)
{
	this->numberOfVoxels = numberOfVoxels;
}

/* OCLArchitecture definition */
OCLArchitecture::OCLArchitecture() : ComputingArchitecture(), xBlockNb(0), yBlockNb(0), zBlockNb(0), xThreadNb(0), yThreadNb(0), zThreadNb(0){}
OCLArchitecture::OCLArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short zBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short zThreadNb) :  ComputingArchitecture(ARCHITECTURE_FPGA, COMPUTE_OCL,gpuNb), xBlockNb(xBlockNb), yBlockNb(yBlockNb), zBlockNb(zBlockNb), xThreadNb(xThreadNb), yThreadNb(yThreadNb), zThreadNb(zThreadNb){}
OCLArchitecture::~OCLArchitecture(){}

OCLArchitecture::OCLArchitecture(const OCLArchitecture & oclArchitectureTocopy)
{
	this->xBlockNb = oclArchitectureTocopy.xBlockNb;
	this->yBlockNb = oclArchitectureTocopy.yBlockNb;
	this->zBlockNb = oclArchitectureTocopy.zBlockNb;
	this->xThreadNb = oclArchitectureTocopy.xThreadNb;
	this->yThreadNb = oclArchitectureTocopy.yThreadNb;
	this->zThreadNb = oclArchitectureTocopy.zThreadNb;
	this->setComputingUnitNb(oclArchitectureTocopy.getComputingUnitNb());
}

OCLArchitecture & OCLArchitecture::operator=(const OCLArchitecture & oclArchitecture)
{
	this->xBlockNb = oclArchitecture.xBlockNb;
	this->yBlockNb = oclArchitecture.yBlockNb;
	this->zBlockNb = oclArchitecture.zBlockNb;
	this->xThreadNb = oclArchitecture.xThreadNb;
	this->yThreadNb = oclArchitecture.yThreadNb;
	this->zThreadNb = oclArchitecture.zThreadNb;
	this->setComputingUnitNb(oclArchitecture.getComputingUnitNb());

	return *this;
}

unsigned short OCLArchitecture::getXBlockNb() const
{
	return xBlockNb;
}

unsigned short OCLArchitecture::getYBlockNb() const
{
	return yBlockNb;
}

unsigned short OCLArchitecture::getZBlockNb() const
{
	return zBlockNb;
}

unsigned short OCLArchitecture::getXThreadNb() const
{
	return xThreadNb;
}

unsigned short OCLArchitecture::getYThreadNb() const
{
	return yThreadNb;
}

unsigned short OCLArchitecture::getZThreadNb() const
{
	return zThreadNb;
}

void OCLArchitecture::setXBlockNb(unsigned short xBlockNb)
{
	this->xBlockNb = xBlockNb;
}

void OCLArchitecture::setYBlockNb(unsigned short yBlockNb)
{
	this->yBlockNb = yBlockNb;
}

void OCLArchitecture::setZBlockNb(unsigned short zBlockNb)
{
	this->zBlockNb = zBlockNb;
}

void OCLArchitecture::setXThreadNb(unsigned short xThreadNb)
{
	this->xThreadNb = xThreadNb;
}

void OCLArchitecture::setYThreadNb(unsigned short yThreadNb)
{
	this->yThreadNb = yThreadNb;
}

void OCLArchitecture::setZThreadNb(unsigned short zThreadNb)
{
	this->zThreadNb = zThreadNb;
}

/* OCLProjectionArchitecture definition */
OCLProjectionArchitecture::OCLProjectionArchitecture() : OCLArchitecture(){}

OCLProjectionArchitecture::OCLProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb) : OCLArchitecture(gpuNb, xBlockNb, yBlockNb, 1, xThreadNb, yThreadNb, 1){}
OCLProjectionArchitecture::~OCLProjectionArchitecture(){}

OCLProjectionArchitecture & OCLProjectionArchitecture::operator=(const OCLProjectionArchitecture & oclProjectionArchitecture)
{
	this->setComputingUnitNb(oclProjectionArchitecture.getComputingUnitNb());
	this->setXBlockNb(oclProjectionArchitecture.getXBlockNb());
	this->setYBlockNb(oclProjectionArchitecture.getYBlockNb());
	this->setXThreadNb(oclProjectionArchitecture.getXThreadNb());
	this->setYThreadNb(oclProjectionArchitecture.getYThreadNb());

	return *this;
}


/* OCLBProjectionArchitecture definition */
OCLBProjectionArchitecture::OCLBProjectionArchitecture() : OCLArchitecture(){}

OCLBProjectionArchitecture::OCLBProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb) : OCLArchitecture(gpuNb, xBlockNb, yBlockNb, 1, xThreadNb, yThreadNb, 1){}
OCLBProjectionArchitecture::~OCLBProjectionArchitecture(){}

OCLBProjectionArchitecture & OCLBProjectionArchitecture::operator=(const OCLBProjectionArchitecture & oclBProjectionArchitecture)
{
	this->setComputingUnitNb(oclBProjectionArchitecture.getComputingUnitNb());
	this->setXBlockNb(oclBProjectionArchitecture.getXBlockNb());
	this->setYBlockNb(oclBProjectionArchitecture.getYBlockNb());
	this->setXThreadNb(oclBProjectionArchitecture.getXThreadNb());
	this->setYThreadNb(oclBProjectionArchitecture.getYThreadNb());

	return *this;
}


///* CUDAArchitecture definition */
//GPUIterativeArchitecture::GPUIterativeArchitecture(CUDAProjectionArchitecture oclprojectionArchitecture, CUDABProjectionArchitecture backProjectionArchitecture){}
//GPUIterativeArchitecture::~GPUIterativeArchitecture(){}


