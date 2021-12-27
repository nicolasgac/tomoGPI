/*
 * GeneralizedGaussianRegularizer_GPU.cuh
 *
 *      Author: gac
 */

#ifndef GENERALIZEDGAUSSIANREGULARIZER_GPU_CUH_
#define GENERALIZEDGAUSSIANREGULARIZER_GPU_CUH_

#include "Regularizer.cuh"
#include "Volume_GPU.cuh"

template<typename T> class GeneralizedGaussianRegularizer_GPU : public Regularizer<Volume_GPU,T>{
public:
	GeneralizedGaussianRegularizer_GPU(double beta);
	~GeneralizedGaussianRegularizer_GPU();

	double getBeta() const;
	void setBeta(double beta);

	void getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda, unsigned int totalIterationIdx, unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;

private:
	double beta;
};

class GeneralizedGaussianRegularizer_GPU_half : public Regularizer_half<Volume_GPU_half>{
public:
	GeneralizedGaussianRegularizer_GPU_half(double beta);
	~GeneralizedGaussianRegularizer_GPU_half();

	double getBeta() const;
	void setBeta(double beta);

	void getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned  int totalIterationIdx,unsigned  int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* betaCG, float lambda) const;

private:
	double beta;
};

#endif /* GENERALIZEDGAUSSIANREGULARIZER_CUH_ */
