/*
 * Image3D_CPU_half.cuh
 *
 *      Author: gac
 */

#ifndef IMAGE3D_CPU_HALF_HPP_
#define IMAGE3D_CPU_HALF_HPP_

#include "Image3D_CPU.cuh"

class Image3D_CPU_half : public Image3D_CPU<half>{
public:
	Image3D_CPU_half();
	Image3D_CPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,CUDAArchitecture* cudaArchitecture);
	Image3D_CPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb);
	~Image3D_CPU_half();

	Image3D_CPU_half(const Image3D_CPU_half& imageToCopy); // Copy-constructor
	Image3D_CPU_half & operator=(const Image3D_CPU_half &image);
	void copyImage3D(Image3D_CPU_half* imageToCopy);
	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);

	void setImage(float value); // Set all pixel of image at value
	void scalarImage(float scalar);// Multiply image by scalar
	void addImage(Image3D_CPU_half* image2); // Add image 2 to current image
	void addImage(Image3D_CPU_half* image2, float lambda); // Add lambda*image2 to current image
	void positiveAddImage(Image3D_CPU_half* image2, float lambda); // Add lambda*image2 to current image with positiveness constraint
	void diffImage(Image3D_CPU_half* image2); // Sub image 2 to current image
	void diffImage(float lambda, Image3D_CPU_half* image2); // Sub image 2 to lambda*(current image)
	void multImage(Image3D_CPU_half* image2); // Multiply current image by image 2
	double scalarProductImage(Image3D_CPU_half* image2);// Scalar product between current image and image 2
	double getImageL1Norm();// Compute L1 Norm of image
	double getImageL2Norm();// Compute L2 Norm of image
	double getImageLpNorm(double p);// Compute Lp Norm of image
	double getImageHuberNorm(double threshold);// Compute Huber Norm of image
	double getImageMean();// Compute mean of image
	double getImageMeanSquare(); // Compute mean square of image
	double getImageStd();// Compute standard deviation of image

	void getImageSign(Image3D_CPU_half* signedImage);// Get signed image
	void getImageAbsPow(Image3D_CPU_half* absPowImage,double p);// Get absolute image raised to pth power

	void saveImage(string fileName); // Save image
	void saveMiddleSliceImage(string fileName); // Save middle slice image
	void loadImage(string fileName); // Load image
private:
	CUDAArchitecture* cudaArchitecture; 
	
};


#endif /* IMAGE3D_HPP_ */
