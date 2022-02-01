/*
 * Volume_CPU.cuh
 *
 *      Author: gac
 */

#ifndef VOLUME_CPU_HPP_
#define VOLUME_CPU_HPP_

#include "Volume.cuh"
#include "Image3D_CPU.cuh"
#include "Image3D_CPU_half.cuh"

template<typename T> class Volume_CPU : public Volume<T>{

public:

	Volume_CPU();
	Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb,T* dataImage);
	Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture);
	Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture,T* dataImage);
	Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb);
	~Volume_CPU();

	Volume_CPU(const Volume_CPU<T>& volumeToCopy);
	//Volume_CPU & operator=(const Volume_CPU<T> &volume);
	Image3D_CPU<T>* getVolumeImage() const;

	void setVolume(T value); // Set all pixel of volume at value
	void scalarVolume(T scalar);// Multiply volume by scalar

	void addVolume(Volume_CPU<T>* volume2); // Add volume 2 to current volume
	void addVolume(Volume_CPU<T>* volume2, T lambda); // Add lambda*volume2 to current volume
	void positiveAddVolume(Volume_CPU<T>* image2, T lambda); // Add lambda*volume2 to current volume with positiveness constraint
	void diffVolume(Volume_CPU<T>* volume2); // Sub volume 2 to current volume
	void diffVolume(T lambda, Volume_CPU<T>* volume2); // Sub volume 2 to lambda*(current volume)
	void diffVolume(Volume_CPU<T>* volume2, T lambda); // Sub lambda*(volume 2) to current volume
	void multVolume(Volume_CPU<T>* volume2); // Multiply current volume by volume 2
	double scalarProductVolume(Volume_CPU<T>* volume2); // Scalar product between current volume and volume 2
	double getVolumeL1Norm(); // // Compute L1 Norm of volume
	double getVolumeL2Norm(); // // Compute L2 Norm of volume
	double getVolumeLpNorm(double p);// Compute Lp Norm of volume
	double getVolumeHuberNorm(double threshold);// Compute Huber Norm of volume
	double getVolumeMean();// Compute mean of volume
	double getVolumeMeanSquare(); // Compute mean square of volume
	double getVolumeStd();// Compute standard deviation of volume

	void getVolumeSign(Volume_CPU<T>* signedVolume);// Get signed volume
	void getVolumeAbsPow(Volume_CPU<T>* absPowVolume, double p);// Get absolute volume raised to pth power

	
	// normalize volume (no centering) by the variances of another volume
	void normalizeVolume(Volume_CPU<T>* volume);
	double sumNormalizedVolume();
	// normalize volume
	void normalizeVolume();
	

	

	// compute the gradient of a volume
	void grad_xplus(Volume_CPU<T>* volume);// (x+1)-x
	void grad_xmoins(Volume_CPU<T>* volume);// x-(x-1)
	void grad_yplus(Volume_CPU<T>* volume);// (y+1)-y
	void grad_ymoins(Volume_CPU<T>* volume);// y-(y-1)
	void grad_zplus(Volume_CPU<T>* volume);// (z+1)-x
	void grad_zmoins(Volume_CPU<T>* volume);// z-(z-1)

	// compute the sign of a volume
	void sign_volume(Volume_CPU<T>* volume);

	// weight volume
	void weightVolume(T* weights);
	double sumWeightedVolume(T* weights);

	
	#ifdef __CUDACC__
	void copyConstantGPU();
#endif
};

class Volume_CPU_half : public Volume_CPU<half>{

public:

	Volume_CPU_half();
	Volume_CPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture);
	~Volume_CPU_half();

	Volume_CPU_half(const Volume_CPU_half& volumeToCopy);
	//Volume_CPU_half & operator=(const Volume_CPU_half &volume);

	Image3D_CPU_half* getVolumeImage() const; // Get volume image

	void setVolume(float value); // Set all pixel of volume at value
	void scalarVolume(float scalar);// Multiply volume by scalar

	void addVolume(Volume_CPU_half* volume2); // Add volume 2 to current volume
	void addVolume(Volume_CPU_half* volume2, float lambda); // Add lambda*volume2 to current volume
	void positiveAddVolume(Volume_CPU_half* image2, float lambda); // Add lambda*volume2 to current volume with positiveness constraint
	void diffVolume(Volume_CPU_half* volume2); // Sub volume 2 to current volume
	void diffVolume(float lambda, Volume_CPU_half* volume2); // Sub volume 2 to lambda*(current volume)
	void multVolume(Volume_CPU_half* volume2); // Multiply current volume by volume 2
	double scalarProductVolume(Volume_CPU_half* volume2); // Scalar product between current volume and volume 2
	double getVolumeL1Norm(); // // Compute L1 Norm of volume
	double getVolumeL2Norm(); // // Compute L2 Norm of volume
	double getVolumeLpNorm(double p);// Compute Lp Norm of volume
	double getVolumeHuberNorm(double threshold);// Compute Huber Norm of volume
	double getVolumeMean();// Compute mean of volume
	double getVolumeMeanSquare(); // Compute mean square of volume
	double getVolumeStd();// Compute standard deviation of volume

	void getVolumeSign(Volume_CPU_half* signedVolume);// Get signed volume
	void getVolumeAbsPow(Volume_CPU_half* absPowVolume, double p);// Get absolute volume raised to pth power

	void saveVolume(string fileName); // Save volume
	void saveMiddleSliceVolume(string fileName); // Save middle slice volume
	void loadVolume(string fileName); // Load volume
	void loadVolume(string fileName,unsigned long int offSet); // Load volume from offSet

	

	// compute the gradient of a volume
	void grad_xplus(Volume_CPU_half* volume);// (x+1)-x
	void grad_xmoins(Volume_CPU_half* volume);// x-(x-1)
	void grad_yplus(Volume_CPU_half* volume);// (y+1)-y
	void grad_ymoins(Volume_CPU_half* volume);// y-(y-1)
	void grad_zplus(Volume_CPU_half* volume);// (z+1)-z
	void grad_zmoins(Volume_CPU_half* volume);// z-(z-1)

	// compute the sign of a volume
	void sign_volume(Volume_CPU_half* volume);


};


#endif /* VOLUME_HPP_ */
