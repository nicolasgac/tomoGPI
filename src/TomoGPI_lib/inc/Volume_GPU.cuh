/*
 * Volume_GPU.cuh
 *
 *      Author: gac
 */

#ifndef VOLUME_GPU_HPP_
#define VOLUME_GPU_HPP_

#include "Volume.cuh"
#include "Image3D_GPU.cuh"


template<typename T> class Volume_GPU : public Volume<T>{

public:

	Volume_GPU();
	Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitectureVolume,T* dataImage);
	Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitectureVolume);
	Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb);
	~Volume_GPU();

	Volume_GPU(const Volume_GPU<T>& volumeToCopy);
	//Volume_GPU & operator=(const Volume_GPU<T> &volume);
	Image3D_GPU<T>* getVolumeImage() const;

	void setVolume(T value); // Set all pixel of volume at value
	void scalarVolume(T scalar);// Multiply volume by scalar

	void addVolume(Volume_GPU<T>* volume2); // Add volume 2 to current volume
	void addVolume(Volume_GPU<T>* volume2, T lambda); // Add lambda*volume2 to current volume
	void positiveAddVolume(Volume_GPU<T>* image2, T lambda); // Add lambda*volume2 to current volume with positiveness constraint
	void diffVolume(Volume_GPU<T>* volume2); // Sub volume 2 to current volume
	void diffVolume(Volume_GPU<T>* volume2, T lambda); // Sub lambda*(volume 2) to current volume
	void diffVolume(T lambda, Volume_GPU<T>* volume2); // Sub volume 2 to lambda*(current volume)
	void multVolume(Volume_GPU<T>* volume2); // Multiply current volume by volume 2
	T scalarProductVolume(Volume_GPU<T>* volume2); // Scalar product between current volume and volume 2
	T getVolumeL1Norm(); // // Compute L1 Norm of volume
	T getVolumeL2Norm(); // // Compute L2 Norm of volume
	T getVolumeLpNorm(T p);// Compute Lp Norm of volume
	T getVolumeHuberNorm(T threshold);// Compute Huber Norm of volume
	T getVolumeMean();// Compute mean of volume
	T getVolumeMeanSquare(); // Compute mean square of volume
	T getVolumeStd();// Compute standard deviation of volume

	void getVolumeSign(Volume_GPU<T>* signedVolume);// Get signed volume
	void getVolumeAbsPow(Volume_GPU<T>* absPowVolume, T p);// Get absolute volume raised to pth power

	// center each pixel and normalize them (MGI) by the means and the variances of another volume
	void normalizeCenterVolumeMGI(Volume_GPU<T>* volume);
	double sumNormalizedCenteredVolumeMGI();
	double normalizeCenterClassesVolumeMGI(Volume_GPU<T>* volume);
	// normalize volume (no centering) by the variances of another volume
	void normalizeVolume(Volume_GPU<T>* volume);
	double sumNormalizedVolume();
	// normalize volume
	void normalizeVolume();
	// center the volume by the means of another volume
	void centerVolumeMGI(Volume_GPU<T>* volume);
	// contours
	void indicateContoursVolume();
	// means for MGM
	void computeMeansMGMVolume(Volume_GPU<T>* volume);

	//Gauss-Markov-Potts : joint MAP (MGI)
	void maxMeansClassesMGI(double m0, double v0);// MAP for m_classes (MGI)
	void maxVariancesClassesMGI(double alpha0, double beta0);//MAP for v_classes (MGI)
	void maxLabelsMGI(unsigned int numit, double tol);//MAP for labels
	void maxLabelsMGIBlancs();//MAP for voxels "blancs"
	void maxLabelsMGINoirs();//MAP for voxels "noirs"
	double computePottsEnergyMGI();//compute Potts energy
	//Gauss-Markov-Potts : joint MAP (MGM)
	void maxMeansClassesMGM(double m0, double v0);// MAP for m_classes (MGM)
	void maxVariancesClassesMGM(double alpha0, double beta0);//MAP for v_classes (MGM)
	void maxMeansClassesMGMKnownContours(double m0, double v0);// MAP for m_classes (MGM) with known contours
	void maxVariancesClassesMGMKnownContours(double alpha0, double beta0);//MAP for v_classes (MGM) with known contours
	void maxLabelsMGM(unsigned int numit, double tol);//MAP for labels (MGM)
	void maxLabelsMGMBlancs();//MAP for voxels "blancs"  (MGM)
	void maxLabelsMGMNoirs();//MAP for voxels "noirs" (MGM)
	double computePottsEnergyMGM();//compute Potts energy (MGM)
	void maxLabelsMGMFixedContours(unsigned int numit, double tol);//MAP for labels (MGM) with fixed contours
	void maxLabelsMGMBlancsFixedContours();//MAP for voxels "blancs"  (MGM) with fixed contours
	void maxLabelsMGMNoirsFixedContours();//MAP for voxels "noirs" (MGM) with fixed contours
	double computePottsEnergyMGMFixedContours();//compute Potts energy (MGM) with fixed contours
	void selectContoursVolume();// select contours of the volume
	void selectNoContoursVolume();// select voxels which are not on the contours
	void selectContoursVolume(Volume_GPU<T>* volume);// select contours of the volume
	void selectNoContoursVolume(Volume_GPU<T>* volume);// select voxels which are not on the contours
	void centerMeanVolume(Volume_GPU<T>* volume);
	double normalizeCenterClassesVolumeMGM(Volume_GPU<T>* volume,Volume_GPU<T>* volume_mgm);
	double sumNormalizedVolumeMGM();
	void centerMeanVolumeKnownContours(Volume_GPU<T>* volume);
	double normalizeCenterClassesVolumeMGMKnownContours(Volume_GPU<T>* volume,Volume_GPU<T>* volume_mgm);
	double normalizeClassesDiffMeansVolume(Volume_GPU<T>* volume, Volume_GPU<T>* volume_means);
	double sumNormalizedVolumeMGMKnownContours();

	// compute the gradient of a volume
	void grad_xplus(Volume_GPU<T>* volume);// (x+1)-x
	void grad_xmoins(Volume_GPU<T>* volume);// x-(x-1)
	void grad_yplus(Volume_GPU<T>* volume);// (y+1)-y
	void grad_ymoins(Volume_GPU<T>* volume);// y-(y-1)
	void grad_zplus(Volume_GPU<T>* volume);// (z+1)-x
	void grad_zmoins(Volume_GPU<T>* volume);// z-(z-1)

	// compute the sign of a volume
	void sign_volume(Volume_GPU<T>* volume);


	// weight volume
	void weightVolume(T* weights);
	double sumWeightedVolume(T* weights);

	//Piece of Nicolas's Program
	void maxLabelsMGI_GPU(unsigned int numit, double tol);
	void maxLabelsMGINoirs_GPU();
	void maxLabelsMGIBlancs_GPU();

private :
	CUDAArchitecture* cudaArchitecture; // GPU architecture for volume
};

class Volume_GPU_half : public Volume_GPU<half>{

public:

	Volume_GPU_half();
	Volume_GPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitectureVolume);
	~Volume_GPU_half();

	Volume_GPU_half(const Volume_GPU_half& volumeToCopy);
	//Volume_GPU_half & operator=(const Volume_GPU_half &volume);

	Image3D_GPU_half* getVolumeImage() const; // Get volume image

	void setVolume(float value); // Set all pixel of volume at value
	void scalarVolume(float scalar);// Multiply volume by scalar

	void addVolume(Volume_GPU_half* volume2); // Add volume 2 to current volume
	void addVolume(Volume_GPU_half* volume2, float lambda); // Add lambda*volume2 to current volume
	void positiveAddVolume(Volume_GPU_half* image2, float lambda); // Add lambda*volume2 to current volume with positiveness constraint
	void diffVolume(Volume_GPU_half* volume2); // Sub volume 2 to current volume
	void diffVolume(float lambda, Volume_GPU_half* volume2); // Sub volume 2 to lambda*(current volume)
	void multVolume(Volume_GPU_half* volume2); // Multiply current volume by volume 2

	//MOdif template <typename P> P en double
	double scalarProductVolume(Volume_GPU_half* volume2); // Scalar product between current volume and volume 2
	double getVolumeL1Norm(); // // Compute L1 Norm of volume
	double getVolumeL2Norm(); // // Compute L2 Norm of volume
	double getVolumeLpNorm(double p);// Compute Lp Norm of volume
	double getVolumeHuberNorm(double threshold);// Compute Huber Norm of volume
	double getVolumeMean();// Compute mean of volume
	double getVolumeMeanSquare(); // Compute mean square of volume
	double getVolumeStd();// Compute standard deviation of volume

	void getVolumeSign(Volume_GPU_half* signedVolume);// Get signed volume
	//MOdif template <typename P> P en double
	 void getVolumeAbsPow(Volume_GPU_half* absPowVolume, double p);// Get absolute volume raised to pth power

	void saveVolume(string fileName); // Save volume
	void saveMiddleSliceVolume(string fileName); // Save middle slice volume
	void loadVolume(string fileName); // Load volume
	void loadVolume(string fileName,unsigned long int offSet); // Load volume from offSet

	// center each pixel and normalize them (MGI) by the means and the variances of another volume
	void normalizeCenterVolumeMGI(Volume_GPU_half* volume);
	double sumNormalizedCenteredVolumeMGI();
	double normalizeCenterClassesVolumeMGI(Volume_GPU_half* volume);
	// normalize volume (no centering) by the variances of another volume
	void normalizeVolume(Volume_GPU_half* volume);
	double sumNormalizedVolume();
	// normalize volume
	void normalizeVolume();
	// center the volume by the means of another volume
	void centerVolumeMGI(Volume_GPU_half* volume);
	// contours
	void indicateContoursVolume();
	// means for MGM
	void computeMeansMGMVolume(Volume_GPU_half* volume);

	//Gauss-Markov-Potts : joint MAP (MGI)
	void maxMeansClassesMGI(double m0, double v0);// MAP for m_classes (MGI)
	void maxVariancesClassesMGI(double alpha0, double beta0);//MAP for v_classes (MGI)
	void maxLabelsMGI(unsigned int numit, double tol);//MAP for labels
	void maxLabelsMGIBlancs();//MAP for voxels "blancs"
	void maxLabelsMGINoirs();//MAP for voxels "noirs"
	double computePottsEnergyMGI();//compute Potts energy
	//Gauss-Markov-Potts : joint MAP (MGM)
	void maxMeansClassesMGM(double m0, double v0);// MAP for m_classes (MGM)
	void maxVariancesClassesMGM(double alpha0, double beta0);//MAP for v_classes (MGM)
	void maxMeansClassesMGMKnownContours(double m0, double v0);// MAP for m_classes (MGM) with known contours
	void maxVariancesClassesMGMKnownContours(double alpha0, double beta0);//MAP for v_classes (MGM) with known contours
	void maxLabelsMGM(unsigned int numit, double tol);//MAP for labels (MGM)
	void maxLabelsMGMBlancs();//MAP for voxels "blancs"  (MGM)
	void maxLabelsMGMNoirs();//MAP for voxels "noirs" (MGM)
	double computePottsEnergyMGM();//compute Potts energy (MGM)
	void maxLabelsMGMFixedContours(unsigned int numit, double tol);//MAP for labels (MGM) with fixed contours
	void maxLabelsMGMBlancsFixedContours();//MAP for voxels "blancs"  (MGM) with fixed contours
	void maxLabelsMGMNoirsFixedContours();//MAP for voxels "noirs" (MGM) with fixed contours
	double computePottsEnergyMGMFixedContours();//compute Potts energy (MGM) with fixed contours
	void selectContoursVolume();// select contours of the volume
	void selectNoContoursVolume();// select voxels which are not on the contours
	void selectContoursVolume(Volume_GPU_half* volume);// select contours of the volume
	void selectNoContoursVolume(Volume_GPU_half* volume);// select voxels which are not on the contours
	void centerMeanVolume(Volume_GPU_half* volume);
	double normalizeCenterClassesVolumeMGM(Volume_GPU_half* volume,Volume_GPU_half* volume_mgm);
	double sumNormalizedVolumeMGM();
	void centerMeanVolumeKnownContours(Volume_GPU_half* volume);
	double normalizeCenterClassesVolumeMGMKnownContours(Volume_GPU_half* volume,Volume_GPU_half* volume_mgm);
	double normalizeClassesDiffMeansVolume(Volume_GPU_half* volume, Volume_GPU_half* volume_means);
	double sumNormalizedVolumeMGMKnownContours();

	// compute the gradient of a volume
	void grad_xplus(Volume_GPU_half* volume);// (x+1)-x
	void grad_xmoins(Volume_GPU_half* volume);// x-(x-1)
	void grad_yplus(Volume_GPU_half* volume);// (y+1)-y
	void grad_ymoins(Volume_GPU_half* volume);// y-(y-1)
	void grad_zplus(Volume_GPU_half* volume);// (z+1)-z
	void grad_zmoins(Volume_GPU_half* volume);// z-(z-1)

	// compute the sign of a volume
	void sign_volume(Volume_GPU_half* volume);


private :
	CUDAArchitecture* cudaArchitecture; // GPU architecture for volume
};


#endif /* VOLUME_HPP_ */
