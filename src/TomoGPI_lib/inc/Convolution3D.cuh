/*
 * Convolution3D.cuh
 *
 *      Author: gac
 */

#ifndef CONVOLUTION3D_HPP_
#define CONVOLUTION3D_HPP_

#include "ieeehalfprecision.hpp"
#include "ComputingArchitecture.cuh"
#include "Volume.cuh"

const int TAILLE_BLOCK 	= 16 ;

#define NUMBER_COMPUTED_BLOCK 8
#define NUMBER_HALO_BLOCK 1
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_H_X 32
#define BLOCK_SIZE_H_Y 32
#define BLOCK_SIZE_V_X 32
#define BLOCK_SIZE_V_Y 32
#define BLOCK_SIZE_P_Z 8
#define BLOCK_SIZE_P_X 32

#define IMUL(a, b) __mul24(a, b)

template<template<typename> class V, typename T> struct TGPUplan_conv3D{
	//! Carte GPU actuelle
	int device;
	//! Le volume d'entrée
	V<T>* volume_in_h;
	//! Le volume de sortie
	V<T>* volume_out_h;
	// Computing Architecture
	int gpuNb;
	cudaStream_t *streams;
	int nstreams;
	unsigned long int zKernelRadius;
};

template <typename V> struct TGPUplan_conv3D_half{
	//! Carte GPU actuelle
	int device;
	//! Le volume d'entrée
	V* volume_in_h;
	//! Le volume de sortie
	V* volume_out_h;
	// Computing Architecture
	int gpuNb;
	cudaStream_t *streams;
	int nstreams;
	unsigned long int zKernelRadius;
};

template<typename T> class Convolution3D{

public:

	Convolution3D(Image3D<T>* kernel); // Constructor for simple 3D convolution
	Convolution3D(T* horizontalKernel, T* verticalKernel, T* depthKernel); // Constructor for separable 3D convolution
	~Convolution3D();

	Image3D<T>* getKernel() const; // Get convolution kernel
	unsigned long int getXKernelRadius() const;
	unsigned long int getYKernelRadius() const;
	unsigned long int getZKernelRadius() const;
	unsigned long int getXKernelSize() const;
	unsigned long int getYKernelSize() const;
	unsigned long int getZKernelSize() const;

	unsigned long int getXFrameSize(Image3D<T>* sourceImage) const;
	unsigned long int getYFrameSize(Image3D<T>* sourceImage) const;
	unsigned long int getZFrameSize(Image3D<T>* sourceImage) const;

#ifdef __CUDACC__
	__host__ void  copyConstantGPU(Volume<T>* sourceImage);
#endif

	//	Image3D<T>* copieFrame3D(Image3D<T>* sourceImage);
	//
	//	/* Simple 3D Convolution on CPU */
	//	void doConvolution3D_CPU(Volume<T>* sourceImage, Volume<T>* convolutedImage);
	//
	//	/* Separable 3D Convolution on CPU */
	//	void doSeparableConvolution3D_CPU(Volume<T>* sourceImage, Volume<T>* convolutedImage);

	/* Separable 3D Convolution on GPU */
	//	void doSeparableConvolution3D_GPU(Volume<T>* sourceImage, Volume<T>* convolutedImage, unsigned long int blockNb);

private:
	Image3D<T>* kernel;
	T* horizontalKernel;
	T* verticalKernel;
	T* depthKernel;
	unsigned long int xKernelRadius;
	unsigned long int yKernelRadius;
	unsigned long int zKernelRadius;
	unsigned long int xKernelSize;
	unsigned long int yKernelSize;
	unsigned long int zKernelSize;
};



#endif /* CONVOLUTION3D_HPP_ */
