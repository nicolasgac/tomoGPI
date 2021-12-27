/*
 * HuberRegularizer_CPU.cuh
 *
 *      Author: gac
 */

#ifndef HUBERREGULARIZER_CPU_CUH_
#define HUBERREGULARIZER_CPU_CUH_

#include "Regularizer.cuh"
#include "Volume_CPU.cuh"

template<typename T> class HuberRegularizer_CPU : public Regularizer<Volume_CPU,T>{

public:

	HuberRegularizer_CPU(double huberThreshold);
	~HuberRegularizer_CPU();

	double getHuberThreshold() const;
	void setHuberThreshold(double huberThreshold);

	void derivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* derivativeHuberVolume) const;
	void secondDerivativeHuberFunction(Volume_CPU<T>* volume, Volume_CPU<T>* secondDerivativeHuberVolume) const;

	void getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;

private:
	double huberThreshold;
};

class HuberRegularizer_CPU_half : public Regularizer_half<Volume_CPU_half>{
public:

	HuberRegularizer_CPU_half(double huberThreshold);
	~HuberRegularizer_CPU_half();

	double getHuberThreshold() const;
	void setHuberThreshold(double huberThreshold);

	void derivativeHuberFunction(Volume_CPU_half* volume, Volume_CPU_half* derivativeHuberVolume) const;
	void secondDerivativeHuberFunction(Volume_CPU_half* volume, Volume_CPU_half* secondDerivativeHuberVolume) const;

	void getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;

private:
	double huberThreshold;
};

#endif /* HUBERREGULARIZER_CUH_ */
