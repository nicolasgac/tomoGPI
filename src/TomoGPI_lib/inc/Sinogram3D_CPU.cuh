/*
 * Sinogram3D_CPU.cuh
 *
  *      Author: gac
 */

#ifndef SINOGRAM3D_CPU_HPP_
#define SINOGRAM3D_CPU_HPP_

#include "Sinogram3D.cuh"



template<typename T> class Sinogram3D_CPU : public Sinogram3D<T>{
public:
	Sinogram3D_CPU();
	Sinogram3D_CPU(const Sinogram3D_CPU& sinogramToCopy); // Copy-constructor
	Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb);
	Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture);
	Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb,T* dataSino);
	Sinogram3D_CPU(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture,T* dataSino);
	~Sinogram3D_CPU();

	void setSinogram(T value); // Set all pixel of sinogram at value
	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);
	void scalarSinogram(T value);// multiply sinogram by a value
	void diffSinogram(Sinogram3D_CPU<T>* sinogram1, Sinogram3D_CPU<T>* sinogram2); // Compute difference between current sinogram and sinogram 2
	void addSinogram(Sinogram3D_CPU<T>* sinogram); // Compute addition of current sinogram by sinogram
	void multSinogram(Sinogram3D_CPU<T>* sinogram); // Compute multiplication of current sinogram by sinogram
	void divideSinogram(Sinogram3D_CPU<T>* sinogram); // Compute division of current sinogram by sinogram
	void weightByVariancesNoise(T* v_noise, int stationnary); // weight current sinogram
	double sumSinogramWeightedL1(T* v_noise, int stationnary); // sum of L1 weighted current sinogram
	double sumSinogramWeightedL2(T* v_noise, int stationnary); // sum of squared weighted current sinogram
	double getSinogramL1Norm();// Compute L1 Norm of sinogram
	double getSinogramL2Norm();// Compute L2 Norm of sinogram
	double getSinogramWeightedL2Norm(T* weights);// Compute weighted L2 norm of sinogram
	double getSinogramLpNorm(double p);// Compute Lp Norm of sinogram
	double getSinogramMean();// Compute mean of sinogram
	double getSinogramMeanSquare(); // Compute mean square of sinogram
	double getSinogramStd();// Compute standard deviation of sinogram
	
	private :
	CUDAArchitecture* cudaArchitecture; 

};

class Sinogram3D_CPU_half : public Sinogram3D_CPU<half>{
public:
	Sinogram3D_CPU_half(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture);
	//Sinogram3D_CPU_half(unsigned long int uSinogramPixelNb, unsigned long int vSinogramPixelNb, unsigned long int projectionSinogramNb, CUDAArchitecture* cudaArchitecture);

	~Sinogram3D_CPU_half();

	Sinogram3D_CPU_half(const Sinogram3D_CPU_half& sinogramToCopy); // Copy-constructor
	Sinogram3D_CPU_half & operator=(const Sinogram3D_CPU_half &sinogram);

	void setSinogram(float value); // Set all pixel of sinogram at value
	void scalarSinogram(float value);//multiply sinogram by a value
	void diffSinogram(Sinogram3D_CPU_half* sinogram1, Sinogram3D_CPU_half* sinogram2); // Compute difference between current sinogram and sinogram 2
	void addSinogram(Sinogram3D_CPU_half* sinogram); // Compute addition of current sinogram by sinogram
	void multSinogram(Sinogram3D_CPU_half* sinogram); // Compute multiplication of current sinogram by sinogram
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
