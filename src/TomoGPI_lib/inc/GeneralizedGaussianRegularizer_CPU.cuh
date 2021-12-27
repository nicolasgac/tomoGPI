/*
 * GeneralizedGaussianRegularizer_CPU.cuh
 *
 *      Author: gac
 */

#ifndef GENERALIZEDGAUSSIANREGULARIZER_CPU_CUH_
#define GENERALIZEDGAUSSIANREGULARIZER_CPU_CUH_

#include "Regularizer.cuh"
#include "Volume_CPU.cuh"

template<typename T> class GeneralizedGaussianRegularizer_CPU : public Regularizer<Volume_CPU,T>{
public:

	GeneralizedGaussianRegularizer_CPU(double beta);
	~GeneralizedGaussianRegularizer_CPU();

	double getBeta() const;
	void setBeta(double beta);

	void getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx, unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, double* jReg, double* normdJProjReg, float lambdaunsigned ,unsigned int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_CPU<T>* volume, Volume_CPU<T>* dJ, Volume_CPU<T>* p, Volume_CPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;

private:
	double beta;
};

class GeneralizedGaussianRegularizer_CPU_half : public Regularizer_half<Volume_CPU_half>{
public:

	GeneralizedGaussianRegularizer_CPU_half(double beta);
	~GeneralizedGaussianRegularizer_CPU_half();

	double getBeta() const;
	void setBeta(double beta);

	void getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_CPU_half* volume, Volume_CPU_half* dJ, Volume_CPU_half* p, Volume_CPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;

private:
	double beta;
};

#endif /* GENERALIZEDGAUSSIANREGULARIZER_CUH_ */
