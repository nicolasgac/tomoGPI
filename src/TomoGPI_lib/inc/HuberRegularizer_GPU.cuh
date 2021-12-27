/*
 * HuberRegularizer_GPU.cuh
 *
 *      Author: gac
 */

#ifndef HUBERREGULARIZER_GPU_CUH_
#define HUBERREGULARIZER_GPU_CUH_

#include "Regularizer.cuh"
#include "Volume_GPU.cuh"

template<typename T> class HuberRegularizer_GPU : public Regularizer<Volume_GPU,T>{
public:
	HuberRegularizer_GPU(double huberThreshold);
	~HuberRegularizer_GPU();

	double getHuberThreshold() const;
	void setHuberThreshold(double huberThreshold);

	void derivativeHuberFunction(Volume_GPU<T>* volume, Volume_GPU<T>* derivativeHuberVolume) const;
	void secondDerivativeHuberFunction(Volume_GPU<T>* volume, Volume_GPU<T>* secondDerivativeHuberVolume) const;

	void getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_GPU<T>* volume, Volume_GPU<T>* dJ, Volume_GPU<T>* p, Volume_GPU<T>* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;

private:
	double huberThreshold;
};

class HuberRegularizer_GPU_half : public Regularizer_half<Volume_GPU_half>{
public:
	HuberRegularizer_GPU_half(double huberThreshold);
	~HuberRegularizer_GPU_half();

	double getHuberThreshold() const;
	void setHuberThreshold(double huberThreshold);

	void derivativeHuberFunction(Volume_GPU_half* volume, Volume_GPU_half* derivativeHuberVolume) const;
	void secondDerivativeHuberFunction(Volume_GPU_half* volume, Volume_GPU_half* secondDerivativeHuberVolume) const;

	void getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda,unsigned int totalIterationIdx,unsigned int optimalStepIterationNb) const;
	void getGradientRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;
	void getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, double* jReg, double* normdJProjReg, float lambda, unsigned int totalIterationIdx, unsigned int optimalStepIterationNb) const;
	void getLaplacianRegularizationCriterion(Volume_GPU_half* volume, Volume_GPU_half* dJ, Volume_GPU_half* p, Volume_GPU_half* olddJ, double* jReg, double* normdJProjReg, double* normdJ, double* normolddJ, double* beta, float lambda) const;

private:
	double huberThreshold;
};

#endif /* HUBERREGULARIZER_CUH_ */
