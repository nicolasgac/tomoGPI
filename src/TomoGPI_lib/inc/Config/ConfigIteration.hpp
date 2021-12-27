/*
 * ConfigIteration.hpp
 *
 *      Author: gac
 */

#ifndef CONFIGITERATION_HPP_
#define CONFIGITERATION_HPP_

#include "Config.hpp"

class ConfigIteration : public Config{

public:

	ConfigIteration();
	ConfigIteration(string configDirectoryName, string configFileName);
	~ConfigIteration();

	unsigned short getAddedNoiseValue(); // Get value of added noise value
	unsigned short getGlobalIterationNb(); // Get number of global iteration
	unsigned short getGradientIterationNb(); // Get number of gradient iteration
	unsigned short getOptimalStepIterationNb(); // Get number of gradient iteration  with optimal step
	double getRegularizationParameter(); // Get regularization parameter
	string getLogFileDirectory(); // Get log file directory
	string getGradient(); // Get gradient type
	string getPenaltyFunction(); // Get penalty function type
	double getHuberThreshold();// Get huber threshold
	double getBeta();// Get beta
	double getGammaPotts(); // get gammaPotts
	unsigned short getSegmentationIterationNb(); // get number of segmentation iterations
	double getM0();
	double getV0();
	double getAlpha0();
	double getBeta0();
	double getAlphaEpsilon0();
	double getBetaEpsilon0();
	int getNumberClasses();
	double getTol();// tolerance to stop JMAP
	string getPottsMGIMGM();// MGI or MGM
	double getTolSegmentation();// tolerance to stop the segmentation
	unsigned short getPositivity();//

	void setAddedNoiseValue(unsigned short addedNoiseValue); // Set value of added noise value
	void setGlobalIterationNb(unsigned short globalIterationNb); // Set number of global iteration
	void setGradientIterationNb(unsigned short gradientIterationNb); // Set number of gradient iteration
	void setOptimalStepIterationNb(unsigned short gradientIterationNb); // Set number of gradient iteration  with optimal step
	void setRegularizationParameter(double regularizationParameter); // Set regularization parameter
	void setLogFileDirectory(string logFileDirectory); // Set log file directory
	void setPenaltyFunction(string penaltyFunction);
	void setGradient(string gradient);
	void setPositivity(char positivity);
	void setHuberThreshold(double huberThreshold);
	void setBeta(double beta);
	void setGammaPotts(double gammaPotts);
	void setSegmentationIterationNb(unsigned short segmentationIterationNb);
	void setM0(double m0);
	void setV0(double v0);
	void setAlpha0(double alpha0);
	void setBeta0(double beta0);
	void setAlphaEpsilon0(double alpha_epsilon0);
	void setBetaEpsilon0(double beta_epsilon0);
	void setNumberClasses(int number_classes);
	void setTol(double tol);//  tolerance to stop JMAP
	void setPottsMGIMGM(string potts_mgi_mgm);// MGI or MGM
	void setTolSegmentation(double tol_seg);// tolerance to stop the segmentation

	void createIterationLogFile(string destinationDirectory);
	void updateLogFile(unsigned short iter, double J, double J_MC, double J_Reg, double alpha, double beta,double eam);

private:
	unsigned short addedNoiseValue; // Value of added noise value
	unsigned short globalIterationNb; // Number of global iteration
	unsigned short gradientIterationNb; // Number of global iteration
	unsigned short optimalStepIterationNb; // Number of gradient iteration with optimal step
	double regularizationParameter; // Regularization parameter
	string logFileDirectory; // Log file directory
	string penaltyFunction; // Type of penalty function
	string gradient; // Type of penalty function
	char positivity;
	double huberThreshold; // Huber Threshold
	double beta; // Value of beta for Lbeta penalty function
	double gammaPotts; // Potts coefficient
	unsigned short segmentationIterationNb; // Number of iterations in segmentation step
	double m0; // mean of the gaussian prior on the means of the classes
	double v0; // variance of the gaussian prior on the means of the classes
	double alpha0; // shape parameter of the inverse gamma prior on the variances of the classes
	double beta0; // intensity parameter of the inverse gamma prior on the variances of the classes
	double alpha_epsilon0; // shape parameter of the inverse gamma prior on the variances of the noise
	double beta_epsilon0; // intensity parameter of the inverse gamma prior on the variances of the noise
	int number_classes; // number of classes
	double tol; //  tolerance to stop JMAP
	double tol_seg;// tolerance to stop the segmentation
	string potts_mgi_mgm;// MGI or MGM

};

#endif /* CONFIGITERATION_HPP_ */
