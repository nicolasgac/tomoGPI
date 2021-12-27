/*
 * Sinogram3D_GPU.cuh
 *
 *      Author: gac
 */

#ifndef SINOGRAM3D_GPU_HPP_
#define SINOGRAM3D_GPU_HPP_


#include "Sinogram3D.cuh"

using namespace std;

class Acquisition;
template <typename T> class Sinogram3D;

template<typename T> class Sinogram3D_GPU : public Sinogram3D<T>{

public:

	Sinogram3D_GPU();
	Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb);
	Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture);
	Sinogram3D_GPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture,T* dataSino);
	~Sinogram3D_GPU();

	Sinogram3D_GPU(const Sinogram3D_GPU& sinogramToCopy); // Copy-constructor

	CUDAArchitecture* getCUDAArchitecture() const;// Get cudaArchitecture pointer
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);// Set cudaArchitecture pointer

	void setSinogram(T value); // Set all pixel of sinogram at value
	void scalarSinogram(T value);// multiply sinogram by a value
	void diffSinogram(Sinogram3D_GPU<T>* sinogram1, Sinogram3D_GPU<T>* sinogram2); // Compute difference between current sinogram and sinogram 2
	void addSinogram(Sinogram3D_GPU<T>* sinogram); // Compute addition of current sinogram by sinogram
	void multSinogram(Sinogram3D_GPU<T>* sinogram); // Compute multiplication of current sinogram by sinogram
	void weightByVariancesNoise(T* v_noise, int stationnary); // weight current sinogram
	double sumSinogramWeightedL1(T* v_noise, int stationnary); // sum of L1 weighted current sinogram
	double sumSinogramWeightedL2(T* v_noise, int stationnary); // sum of squared weighted current sinogram
	T getSinogramL1Norm();// Compute Lp Norm of sinogram
	T getSinogramL2Norm();// Compute Lp Norm of sinogram
	T getSinogramLpNorm(T p);// Compute Lp Norm of sinogram
	T getSinogramMean();// Compute mean of sinogram
	T getSinogramMeanSquare(); // Compute mean square of sinogram
	T getSinogramStd();// Compute standard deviation of sinogram
	void saveSinogram(string fileName);

	unsigned int nextPow2(unsigned int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}


#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif

private:
	
	CUDAArchitecture* cudaArchitecture; // GPU architecture for sinogram
};

class Sinogram3D_GPU_half : public Sinogram3D_GPU<half>{

public:

	Sinogram3D_GPU_half(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture);
	~Sinogram3D_GPU_half();

	Sinogram3D_GPU_half(const Sinogram3D_GPU_half& sinogramToCopy); // Copy-constructor
	Sinogram3D_GPU_half & operator=(const Sinogram3D_GPU_half &sinogram);

	void setSinogram(float value); // Set all pixel of sinogram at value
	void scalarSinogram(float value);//multiply sinogram by a value
	void diffSinogram(Sinogram3D_GPU_half* sinogram1, Sinogram3D_GPU_half* sinogram2); // Compute difference between current sinogram and sinogram 2
	void addSinogram(Sinogram3D_GPU_half* sinogram); // Compute addition of current sinogram by sinogram
	void multSinogram(Sinogram3D_GPU_half* sinogram); // Compute multiplication of current sinogram by sinogram
	double getSinogramL1Norm();// Compute Lp Norm of sinogram
	double getSinogramL2Norm();// Compute Lp Norm of sinogram
	double getSinogramLpNorm(double p);// Compute Lp Norm of sinogram
	double getSinogramMean();// Compute mean of sinogram
	double getSinogramMeanSquare(); // Compute mean square of sinogram
	double getSinogramStd();// Compute standard deviation of sinogram

	void saveSinogram(string fileName); // Save sinogram
	void loadSinogram(string fileName); // Load sinogram
};


#endif /* SINOGRAM3D_HPP_ */
