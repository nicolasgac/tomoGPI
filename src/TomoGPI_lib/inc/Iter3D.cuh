/*
 * iter3d.cuh
 *
 *      Author: gac
 */

#ifndef ITER3D_HPP_
#define ITER3D_HPP_

#ifdef __linux__
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#endif
#include <omp.h>

//#include "mex.h"
//#include "class_handle_iter.hpp"

#include "Acquisition.hpp"
#include "GPUConstant.cuh"
#include "ComputingArchitecture.cuh"
#include "ConfigCT.hpp"
#include "ConfigComputeArchitecture.hpp"
#include "ConfigIteration.hpp"
#include "ConfigTiff.hpp"
#include "Convolution3D.cuh"
#include "Detector.hpp"
#include "FieldOfView.hpp"
#include "Image3D.cuh"
#include "GeneralizedGaussianRegularizer_CPU.cuh"
#include "GeneralizedGaussianRegularizer_GPU.cuh"
//#include "GeneralizedGaussianRegularizer_MGPU.cuh"
#include "HuberRegularizer_CPU.cuh"
#include "HuberRegularizer_GPU.cuh"
//#include "HuberRegularizer_MGPU.cuh"
#include "ieeehalfprecision.hpp"
#include "Regularizer.cuh"
#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Sinogram3D.cuh"
#include "Sinogram3D_CPU.cuh"
#include "Sinogram3D_GPU.cuh"
//#include "Sinogram3D_MGPU.cuh"

#include "BackProjector.cuh"
#include "BackProjector_CPU.cuh"
#include "BackProjector_GPU.cuh"
//#include "BackProjector_MGPU.cuh"
#include "Projector.cuh"
#include "Projector_CPU.cuh"
#include "Projector_GPU.cuh"
//#include "Projector_MGPU.cuh"

typedef enum {HUBER,GG}kind_reg;
typedef enum {PROJBACK,SIMPLE,CONJUGATE}kind_gradient;

//#define COMPUTE_J 1
//#define COMPUTE_EAM 1
//#define COMPUTE_MIDDLE_SLICE 1
#define TOMOGPI_VERSION "1.0"

/*typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_CPU;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_GPU;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_CPU_half;
typedef union {HuberRegularizer_CPU,GeneralizedGaussianRegularizer_CPU} type_reg_GPU_half;*/

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG, template<typename> class C, template<typename> class V, template<typename> class S, typename T> class Iter3D{

public:
	Iter3D(string workdirectory);
	Iter3D(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file);

	virtual ~Iter3D();
	virtual V<T>* create_volume();
	virtual S<T>* create_sinogram3D();
	virtual void Init1_SG(V<T>* volume,S<T>* sino);
	virtual void Init2_SG(V<T>* volume,V<T>* dJ,S<T>* sino);
	virtual void Init3_SG(S<T>* sino);
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;
	string getWorkDirectory() const;
	string getSinogramDirectory() const;
	string getOutputDirectory() const;

	ConfigCT* getConfigCTFile() const;
	ConfigComputeArchitecture* getConfigComputeArchitectureFile() const;
	ConfigTiff<unsigned short>* getConfigTiffFile() const;
	Config* getConfigProjection() const;
	Config* getConfigProjectionScheme() const;
	ConfigIteration *getConfigIterationFile() const;
	Acquisition* getAcquisition() const;
	Detector* getDetector() const;
	CylindricFOV * getFieldOfview() const;

	V<T>* getVolume() const;
	void setVolume(V<T>*  volume) ;
	BP<T>* getBackprojector() const;
	void setBackprojector(BP<T>*  proj) ;
	P<T>* getProjector() const;
	void setProjector(P<T>*  proj) ;
	kind_reg getKind_Regularizer() const ;
	void setKind_Regularizer(kind_reg reg)  ;
	kind_gradient getKind_Gradient() const ;
	void setKind_Gradient(kind_gradient gradient)  ;
	/*R_Huber<T>* getRegularizer_Huber() const;
		void setRegularizer_Huber(R_Huber<T>*  reg) ;*/
	R_Huber<T>* getRegularizer_Huber() const;
	void setRegularizer_Huber(R_Huber<T>*  reg) ;
	R_GG<T>* getRegularizer_GG() const;
	void setRegularizer_GG(R_GG<T>*  reg) ;
	double getLambda() const;
	char getPositivity() const;
	double getNoiseValue() const;
	unsigned int getGlobalIterationNb() const;
	unsigned int getGradientIterationNb() const;
	unsigned int getOptimalStepIterationNb() const;
	void setLambda(double lambda) ;
	void setPositivity(char positivity);
	void setNoiseValue(double NoiseValue) ;
	void setGlobalIterationNb(unsigned int globalIterationNb) ;
	void setGradientIterationNb(unsigned int gradientIterationNb) ;
	void setOptimalStepIterationNb(unsigned int optimalStepIterationNb);
	void doProjBack(V<T>* volume);
	void doSimpleGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume);
	void doWeightedGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume);
	void doConjugateGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume);


	// total variation
	double* doPrimalDualFrankWolfeTV(V<T>* volume,S<T>* realSinogram, S<T>* dual_proj, V<T>* dual_vol, T* v_noise, double lambda, double norm_H_grad,int stationnary, unsigned int numit);

	int doMainIter( int argc, char** argv);

private:
	string workDirectory;
	string sinogramDirectory;
	string outputDirectory;
	kind_reg reg;
	kind_gradient gradient;
	char positivity;
	ConfigCT* configCTFile; // CT GEOMETRY CONFIG FILE
	ConfigIteration *configIterationFile; // ITERATIVE ALGORITHM CONFIG FILE
	ConfigComputeArchitecture *configComputeArchitectureFile; // COMPUTE ARCHITECTURE CONFIG FILE
	ConfigTiff<unsigned short> *configTiffFile; // TIFF CONFIG FILE
	Config *configProjection; // PROJECTION CONFIG FILE
	Config *configProjectionScheme; // PROJECTIONSCHEME CONFIG FILE

	Detector *detector;
	CylindricFOV *fieldOfview;
	Acquisition *acquisition;

	P<T> *proj;
	BP<T> *backproj;
	V<T>* volume;

	R_Huber<T> *reg_Huber;
	R_GG<T> *reg_GG;

	double lambda;
	double noiseValue;
	unsigned int globalIterationNb;
	unsigned int gradientIterationNb;
	unsigned int optimalStepIterationNb;
};

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG, template<typename> class C, template<typename> class V, template<typename> class S, typename T> class Iter3D_compute_CUDA : public Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>{
public:

	Iter3D_compute_CUDA(string workdirectory);
	Iter3D_compute_CUDA(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file);

	~Iter3D_compute_CUDA();
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;
	V<T>* create_volume();
	S<T>* create_sinogram3D();
	void Init1_SG(V<T>* volume,S<T>* sino);
	void Init2_SG(V<T>* volume,V<T>* dJ,S<T>* sino);
	void Init3_SG(S<T>* sino);

	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const;
	CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const;
	CUDAArchitecture* getCUDAArchitectureSino() const;
	CUDAArchitecture* getCUDAArchitectureVolume() const;
	CUDAArchitecture* getCUDAArchitecture() const;
#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif

private:
	CUDAArchitecture *cudaArchitecture;
	CUDAArchitecture *cudaArchitectureSino;
	CUDAArchitecture *cudaArchitectureVolume;
	CUDAProjectionArchitecture *cudaprojectionArchitecture;
	CUDABProjectionArchitecture *cudabackprojectionArchitecture;
};

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG, template<typename> class C, template<typename> class V, template<typename> class S, typename T> class Iter3D_compute_C : public Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>{
public:
	Iter3D_compute_C(string workdirectory);
	Iter3D_compute_C(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file);
	V<T>* create_volume();
	S<T>* create_sinogram3D();
	~Iter3D_compute_C();
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;

private:
	CUDAArchitecture *cudaArchitectureSino;
};


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG, template<typename> class C, template<typename> class V, template<typename> class S, typename T> class Iter3D_compute_OCL : public Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>{
public:
	Iter3D_compute_OCL(string workdirectory);
	Iter3D_compute_OCL(string workdirectory, ConfigComputeArchitecture* configComputeArchitecture_file);
	V<T>* create_volume();
	S<T>* create_sinogram3D();
	~Iter3D_compute_OCL();
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;
	OCLProjectionArchitecture* getOCLProjectionArchitecture() const;
	OCLBProjectionArchitecture* getOCLBProjectionArchitecture() const;
	OCLArchitecture* getOCLArchitectureSino() const;
	OCLArchitecture* getOCLArchitectureVolume() const;
	OCLArchitecture* getOCLArchitecture() const;

	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const;
	CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const;
	CUDAArchitecture* getCUDAArchitectureSino() const;
	CUDAArchitecture* getCUDAArchitectureVolume() const;
	CUDAArchitecture* getCUDAArchitecture() const;
	//kind_architecture* getArchitecture_kind() const;

private:
	OCLArchitecture *oclArchitecture;
	OCLArchitecture *oclArchitectureSino;
	OCLArchitecture *oclArchitectureVolume;
	OCLProjectionArchitecture *oclprojectionArchitecture;
	OCLBProjectionArchitecture *oclbackprojectionArchitecture;

	CUDAArchitecture *cudaArchitecture;
	CUDAArchitecture *cudaArchitectureSino;
	CUDAArchitecture *cudaArchitectureVolume;
	CUDAProjectionArchitecture *cudaprojectionArchitecture;
	CUDABProjectionArchitecture *cudabackprojectionArchitecture;
	kind_architecture oclArchitecture_kind;
};

template<typename P, typename BP, typename R_Huber, typename R_GG, typename C, typename V, typename S> class Iter3D_half{
public:
	Iter3D_half(string workdirectory);
	Iter3D_half(string workdirectory,ConfigComputeArchitecture *configComputeArchitectureFile);
	virtual ~Iter3D_half();
	//virtual void doProjection(S* estimatedSinogram) = 0;
	//virtual void doProjection(S<T>* estimatedSinogram,V<T>*volume) = 0;
	string getWorkDirectory() const;
	string getSinogramDirectory() const;
	string getOutputDirectory() const;

	ConfigCT* getConfigCTFile() const;
	ConfigComputeArchitecture* getConfigComputeArchitectureFile() const;
	ConfigTiff<unsigned short>* getConfigTiffFile() const;
	Config* getConfigProjection() const;
	Config* getConfigProjectionScheme() const;
	ConfigIteration *getConfigIterationFile() const;

	Acquisition* getAcquisition() const;
	Detector* getDetector() const;
	CylindricFOV * getFieldOfview() const;

	CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const;
	CUDABProjectionArchitecture* getCUDABProjectionArchitecture() const;
	CUDAArchitecture* getCUDAArchitectureSino() const;
	CUDAArchitecture* getCUDAArchitectureVolume() const;

	V* getVolume() const;
	void setVolume(V*  volume) ;
	BP* getBackprojector() const;
	void setBackprojector(BP*  proj) ;
	P* getProjector() const;
	void setProjector(P*  proj) ;
	kind_reg getKind_Regularizer() const ;
	void setKind_Regularizer(kind_reg reg)  ;
	kind_gradient getKind_Gradient() const ;
	void setKind_Gradient(kind_gradient gradient)  ;

	/*R_Huber<T>* getRegularizer_Huber() const;
			void setRegularizer_Huber(R_Huber<T>*  reg) ;*/

	R_Huber* getRegularizer_Huber() const;
	void setRegularizer_Huber(R_Huber*  reg) ;

	R_GG* getRegularizer_GG() const;
	void setRegularizer_GG(R_GG*  reg) ;
	double getLambda() const;
	double getNoiseValue() const;
	unsigned int getGlobalIterationNb() const;
	unsigned int getGradientIterationNb() const;
	unsigned int getOptimalStepIterationNb() const;
	void setLambda(double lambda) ;
	char getPositivity() const;
	void setPositivity(char positivity) ;
	void setNoiseValue(double noiseValue) ;
	void setGlobalIterationNb(unsigned int globalIterationNb) ;
	void setGradientIterationNb(unsigned int gradientIterationNb) ;
	void setOptimalStepIterationNb(unsigned int optimalStepIterationNb) ;

	void doSimpleGradient(V* volume,S* realSinogram,V* realVolume);
	void doWeightedGradient(V* volume,S* realSinogram,V* realVolume);
	void doConjugateGradient(V* volume,S* realSinogram,V* realVolume);

	int doMainIter( int argc, char** argv);
	//void doMexIter( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[],char *cmd);
private:
	string workDirectory;
	string sinogramDirectory;
	string outputDirectory;
	kind_reg reg;
	kind_gradient gradient;
	char positivity;
	ConfigCT* configCTFile; // CT GEOMETRY CONFIG FILE
	ConfigIteration *configIterationFile; // ITERATIVE ALGORITHM CONFIG FILE
	ConfigComputeArchitecture *configComputeArchitectureFile; // COMPUTE ARCHITECTURE CONFIG FILE
	ConfigTiff<unsigned short> *configTiffFile; // TIFF CONFIG FILE
	Config *configProjection; // PROJECTION CONFIG FILE
	Config *configProjectionScheme; // PROJECTIONSCHEME CONFIG FILE
	Detector *detector;
	CylindricFOV *fieldOfview;
	Acquisition *acquisition;
	CUDAArchitecture *cudaArchitectureSino;
	CUDAArchitecture *cudaArchitectureVolume;
	CUDAProjectionArchitecture *cudaprojectionArchitecture;
	CUDABProjectionArchitecture * cudabackprojectionArchitecture;
	P *proj;
	BP *backproj;
	V* volume;
	R_Huber *reg_Huber;
	R_GG *reg_GG;
	double lambda;
	double noiseValue;
	unsigned int globalIterationNb;
	unsigned int gradientIterationNb;
	unsigned int optimalStepIterationNb;
};

#endif /* ITER3D_HPP_ */
