/*
 * Image3D_CPU.cuh
 *
 *      Author: gac
 */

#ifndef IMAGE3D_CPU_HPP_
#define IMAGE3D_CPU_HPP_

#include "Image3D.cuh"

template<typename T> class Image3D_CPU : public Image3D<T>{
public:
	Image3D_CPU();
	Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,T* dataImage);
	Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,CUDAArchitecture* cudaArchitecture);
	Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,CUDAArchitecture* cudaArchitecture,T* dataImage);
	Image3D_CPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb);
	~Image3D_CPU();

	Image3D_CPU(const Image3D_CPU<T>& imageToCopy); // Copy-constructor
	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);

	void copyImage3D(Image3D_CPU<T>* imageToCopy);
	void setImage(T value); // Set all pixel of image at value
	void scalarImage(T scalar);// Multiply image by scalar
	void addImage(Image3D_CPU<T>* image2); // Add image 2 to current image
	void addImage(Image3D_CPU<T>* image2, T lambda); // Add lambda*image2 to current image
	void positiveAddImage(Image3D_CPU<T>* image2,T lambda); // Add lambda*image2 to current image with positiveness constraint
	void diffImage(Image3D_CPU<T>* image2); // Sub image 2 to current image
	void diffImage(T lambda, Image3D_CPU<T>* image2); // Sub image 2 to lambda*(current image)
	void diffImage(Image3D_CPU<T>* image2, T lambda); // Sub lambda*(image 2) to current image
	void multImage(Image3D_CPU<T>* image2); // Multiply current image by image 2
	double scalarProductImage(Image3D_CPU<T>* image2);// Scalar product between current image and image 2
	double getImageL1Norm();// Compute L1 Norm of image
	double getImageL2Norm();// Compute L2 Norm of image
	double getImageLpNorm(double p);// Compute Lp Norm of image
	double getImageHuberNorm(double threshold);// Compute Huber Norm of image
	double getImageMean();// Compute mean of image
	double getImageMeanSquare(); // Compute mean square of image
	double getImageStd();// Compute standard deviation of image

	void getImageSign(Image3D_CPU<T>* signedImage);// Get signed image
	void getImageAbsPow(Image3D_CPU<T>* absPowImage,double p);// Get absolute image raised to pth power

	inline double fast_sign(double f) {
		if (f > 0) return 1.0;
		return (f == 0) ? 0.0 : -1.0;
	}
	private :
	CUDAArchitecture* cudaArchitecture; 
	
};


template<typename T> class ImageTiff : public Image3D_CPU<T>{

public:

	ImageTiff(unsigned short xImagePixelNb, unsigned short yImagePixelNb);
	~ImageTiff();

	void loadTiffImage(string fileName); // Load tiff image

};

template<typename T> class ImageCalibration : public ImageTiff<T>{

public:

	ImageCalibration(unsigned short xImagePixelNb, unsigned short yImagePixelNb);
	~ImageCalibration();

	double getBlackMeanValue(); // Get black mean value of calibration image
	double getGrayMeanValue(); // Get gray mean value of calibration image
	double getWhiteMeanValue(unsigned int CalibrationEnergyNb); // Get white mean value of calibration image

	ImageTiff<T>& getBlackTiffImage(); // Get black tiff image
	ImageTiff<T>& getGrayTiffImage(); // Get black tiff image
	ImageTiff<T>& getWhiteTiffImage(); // Get black tiff image

};

#endif /* IMAGE3D_HPP_ */
