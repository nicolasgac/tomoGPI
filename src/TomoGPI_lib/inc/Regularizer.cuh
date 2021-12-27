/*
 * Regularizer.cuh
 *
 *      Author: gac
 */

#ifndef REGULARIZER_HPP_
#define REGULARIZER_HPP_

#ifdef __linux__ 
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "Volume.cuh"
#include "Volume_CPU.cuh"
#include "Volume_GPU.cuh"
//#include "Volume_MGPU.cuh"
#include "Convolution3D.cuh"
#include "Convolution3D_CPU.cuh"
#include "Convolution3D_GPU.cuh"
//#include "Convolution3D_MGPU.cuh"

template<template<typename> class V, typename T>  class Regularizer{

public:
	Regularizer();
	virtual ~Regularizer();

	virtual void getGradientRegularizationCriterion(V<T>* volume, V<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const = 0; // Get criterion of regularization based on gradient of image
	virtual void getGradientRegularizationCriterion(V<T>* volume, V<T>* dJ, V<T>* p, V<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const = 0; // Get criterion of regularization based on gradient of image for Conjugate Gradient Descent
	virtual void getLaplacianRegularizationCriterion(V<T>* volume, V<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx, unsigned int optimalStepIterationNb) const = 0; // Get criterion of regularization based on laplacian of image
	virtual void getLaplacianRegularizationCriterion(V<T>* volume, V<T>* dJ, V<T>* p, V<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const = 0; // Get criterion of regularization based on laplacian of image for Conjugate Gradient Descent
};

template <typename V>  class Regularizer_half{
public:
	Regularizer_half();
	virtual ~Regularizer_half();

	virtual void getGradientRegularizationCriterion(V* volume, V* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned  int optimalStepIterationNb) const = 0; // Get criterion of regularization based on gradient of image
	virtual void getGradientRegularizationCriterion(V* volume, V* dJ, V* p, V* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const = 0; // Get criterion of regularization based on gradient of image for Conjugate Gradient Descent
	virtual void getLaplacianRegularizationCriterion(V* volume, V* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned  int optimalStepIterationNb) const = 0; // Get criterion of regularization based on laplacian of image
	virtual void getLaplacianRegularizationCriterion(V* volume, V* dJ, V* p, V* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const = 0; // Get criterion of regularization based on laplacian of image for Conjugate Gradient Descent
};

#endif /* REGULARIZER_HPP_ */
