/*
 * ComputingArchitecture.cuh
 *
 *      Author: gac
 */

#ifndef COMPUTINGARCHITECTURE_HPP_
#define COMPUTINGARCHITECTURE_HPP_
#include <helper_functions.h>
#include <helper_cuda.h>
#include <multithreading.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <CL/opencl.h>



#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>


typedef enum {ARCHITECTURE_FPGA,ARCHITECTURE_GPU,ARCHITECTURE_CPU}kind_architecture;
typedef enum {HALF_GPU,FLOAT_GPU,DOUBLE_GPU}kind_precision;
typedef enum {MEM_CPU,MEM_GPU, MEM_MGPU}kind_storage;
typedef enum {PAIR_RSVI,PAIR_SFTR, PAIR_SIDDONVI}kind_pair;
typedef enum {COMPUTE_CUDA,COMPUTE_C,COMPUTE_OCL}kind_compute;

const int MAX_GPU_COUNT = 8;
extern std::mutex iomutex;
extern std::atomic<int> counter ;



#ifdef __CUDACC__
template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
    																		{
		extern __shared__ int __smem[];
		return (T *)__smem;
    																		}

	__device__ inline operator const T *() const
    																		{
		extern __shared__ int __smem[];
		return (T *)__smem;
    																		}
};
#endif

class ComputingArchitecture{

public:

	ComputingArchitecture();
	ComputingArchitecture(kind_architecture architecture, kind_compute compute,  unsigned short computingUnitNb);
	~ComputingArchitecture();

	ComputingArchitecture & operator=(const ComputingArchitecture &computingArchitecture);

	unsigned short getComputingUnitNb() const; // Get number of computing units
	void setComputingUnitNb(unsigned short ComputingUnitNb); // Set number of computing units

	kind_architecture getArchitecture() const; // 
	void setArchitecture(kind_architecture architecture); // 

	kind_compute getCompute() const; // 
	void setCompute(kind_compute compute); // 


private:
	kind_architecture architecture;
	kind_compute compute;
	unsigned short computingUnitNb;
	

};


class CUDAArchitecture : public ComputingArchitecture{

public:

	CUDAArchitecture();
	CUDAArchitecture(unsigned short gpuNb,unsigned short xBlockNb, unsigned short yBlockNb, unsigned short zBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short zThreadNb);
	CUDAArchitecture(const CUDAArchitecture &cudaArchitectureToCopy);
	~CUDAArchitecture();

	CUDAArchitecture & operator=(const CUDAArchitecture &cudaArchitecture);

	unsigned short getXBlockNb() const; // Get X block number
	unsigned short getYBlockNb() const; // Get Y block number
	unsigned short getZBlockNb() const; // Get Z block number
	unsigned short getXThreadNb() const; // Get X thread number
	unsigned short getYThreadNb() const; // Get Y thread number
	unsigned short getZThreadNb() const; // Get Z thread number


	void setXBlockNb(unsigned short xBlockNb); // Set X block number
	void setYBlockNb(unsigned short yBlockNb); // Set Y block number
	void setZBlockNb(unsigned short yBlockNb); // Set Z block number
	void setXThreadNb(unsigned short xThreadNb); // Set X thread number
	void setYThreadNb(unsigned short xThreadNb); // Set Y thread number
	void setZThreadNb(unsigned short xThreadNb); // Set Z thread number

private:
	unsigned short xBlockNb;
	unsigned short yBlockNb;
	unsigned short zBlockNb;
	unsigned short xThreadNb;
	unsigned short yThreadNb;
	unsigned short zThreadNb;





};

class CUDAProjectionArchitecture : public CUDAArchitecture{
public:

	CUDAProjectionArchitecture();
	CUDAProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short projectionThreadNb, unsigned short projectionStreamsNb, unsigned short numberOfCells);
	~CUDAProjectionArchitecture();

	CUDAProjectionArchitecture & operator=(const CUDAProjectionArchitecture &cudaProjectionArchitecture);

	unsigned short getProjectionThreadNb() const; // Get projection thread number
	unsigned short getProjectionStreamsNb() const;// Get Streams Number for projection
	unsigned short getNumberOfCells() const;

	void setProjectionThreadNb(unsigned short projectionThreadNb); // Set projection thread number
	void setProjectionStreamsNb(unsigned short projectionStreamsNb);// Set Projection Streams Number for projection
	void setNumberOfCells(unsigned short numberOfCells);

private:
	unsigned short projectionThreadNb; // Projection thread number
	unsigned short projectionStreamsNb; // Projection thread number
	unsigned short numberOfCells;
};

class CUDABProjectionArchitecture : public CUDAArchitecture{
public:

	CUDABProjectionArchitecture();
	CUDABProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short projectionLoopSize, unsigned short bprojectionStreamsNb, unsigned short numberOfCells);
	~CUDABProjectionArchitecture();

	CUDABProjectionArchitecture & operator=(const CUDABProjectionArchitecture &cudaBProjectionArchitecture);

	unsigned short getProjectionLoopSize() const;// Get Projection loop size for backprojection
	unsigned short getBProjectionStreamsNb() const;// Get Streams Number for backprojection
	unsigned short getNumberOfVoxels() const;

	void setProjectionLoopSize(unsigned short bprojectionStreamsNb);// Set Projection loop size for backprojection
	void setBProjectionStreamsNb(unsigned short bprojectionStreamsNb);// Set Projection Streams Number for backprojection
	void setNumberOfVoxels(unsigned short numberOfVoxels);

private:
	unsigned short projectionLoopSize;// Projection loop size for backprojection
	unsigned short bprojectionStreamsNb;
	unsigned short numberOfVoxels;
};

class OCLArchitecture : public ComputingArchitecture{
public:

	OCLArchitecture();
	OCLArchitecture(unsigned short gpuNb,unsigned short xBlockNb, unsigned short yBlockNb, unsigned short zBlockNb, unsigned short xThreadNb, unsigned short yThreadNb, unsigned short zThreadNb);
	OCLArchitecture(const OCLArchitecture &oclArchitectureToCopy);
	~OCLArchitecture();

	OCLArchitecture & operator=(const OCLArchitecture &oclArchitecture);

	unsigned short getXBlockNb() const; // Get X block number
	unsigned short getYBlockNb() const; // Get Y block number
	unsigned short getZBlockNb() const; // Get Z block number
	unsigned short getXThreadNb() const; // Get X thread number
	unsigned short getYThreadNb() const; // Get Y thread number
	unsigned short getZThreadNb() const; // Get Z thread number

	void setXBlockNb(unsigned short xBlockNb); // Set X block number
	void setYBlockNb(unsigned short yBlockNb); // Set Y block number
	void setZBlockNb(unsigned short yBlockNb); // Set Z block number
	void setXThreadNb(unsigned short xThreadNb); // Set X thread number
	void setYThreadNb(unsigned short xThreadNb); // Set Y thread number
	void setZThreadNb(unsigned short xThreadNb); // Set Z thread number

private:
	unsigned short xBlockNb;
	unsigned short yBlockNb;
	unsigned short zBlockNb;
	unsigned short xThreadNb;
	unsigned short yThreadNb;
	unsigned short zThreadNb;
};

class OCLProjectionArchitecture : public OCLArchitecture{
public:

	OCLProjectionArchitecture();
	OCLProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb);
	~OCLProjectionArchitecture();

	OCLProjectionArchitecture & operator=(const OCLProjectionArchitecture &oclProjectionArchitecture);

private:
};

class OCLBProjectionArchitecture : public OCLArchitecture{
public:

	OCLBProjectionArchitecture();
	OCLBProjectionArchitecture(unsigned short gpuNb, unsigned short xBlockNb, unsigned short yBlockNb, unsigned short xThreadNb, unsigned short yThreadNb);
	~OCLBProjectionArchitecture();

	OCLBProjectionArchitecture & operator=(const OCLBProjectionArchitecture &oclBProjectionArchitecture);

private:
};


#endif /* COMPUTINGARCHITECTURE_HPP_ */
