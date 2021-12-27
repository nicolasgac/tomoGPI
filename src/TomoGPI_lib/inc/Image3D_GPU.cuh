/*
 * Image3D_GPU.cuh
 *
 *      Author: gac
 */

#ifndef IMAGE3D_GPU_HPP_
#define IMAGE3D_GPU_HPP_

#include "Image3D.cuh"

using namespace std;


template<typename T>class Image3D_GPU : public Image3D<T>{

public:

	Image3D_GPU();
	Image3D_GPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitectureVolume);
	~Image3D_GPU();

	Image3D_GPU(const Image3D_GPU<T>& imageToCopy); // Copy-constructor

	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);

	void copyImage3D(Image3D_GPU<T>* imageToCopy);
	void scalarImage(T scalar);// Multiply image by scalar
	void setImage(T value); // Set all pixel of image at value
	void addImage(Image3D_GPU<T>* image2); // Add image 2 to current image
	void addImage(Image3D_GPU<T>* image2, T lambda); // Add lambda*image2 to current image
	void positiveAddImage(Image3D_GPU<T>* image2, T lambda); // Add lambda*image2 to current image with positiveness constraint
	void diffImage(Image3D_GPU<T>* image2); // Sub image 2 to current image
	void diffImage(T lambda, Image3D_GPU<T>* image2); // Sub image 2 to lambda*(current image)
	void diffImage(Image3D_GPU<T>* image2, T lambda); // Sub lambda*(image 2) to current image
	void multImage(Image3D_GPU<T>* image2); // Multiply current image by image 2
	T scalarProductImage(Image3D_GPU<T>* image2);// Scalar product between current image and image 2
	T getImageL1Norm();// Compute L1 Norm of image
	T getImageL2Norm();// Compute L2 Norm of image
	T getImageLpNorm(T p);// Compute Lp Norm of image
	T getImageHuberNorm(T threshold);// Compute Huber Norm of image
	T getImageMean();// Compute mean of image
	T getImageMeanSquare(); // Compute mean square of image
	T getImageStd();// Compute standard deviation of image

	void getImageSign(Image3D_GPU<T>* signedImage);// Get signed image
	void getImageAbsPow(Image3D_GPU<T>* absPowImage,T p);// Get absolute image raised to pth power

	unsigned int nextPow2(unsigned int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}

#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif
private:
	CUDAArchitecture* cudaArchitecture; // GPU architecture for sinogram
	

};

class Image3D_GPU_half : public Image3D_GPU<half>{

public:

	Image3D_GPU_half();
	Image3D_GPU_half(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitectureVolume);
	~Image3D_GPU_half();

	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);

	Image3D_GPU_half(const Image3D_GPU_half& imageToCopy); // Copy-constructor
	Image3D_GPU_half & operator=(const Image3D_GPU_half &image);
	void copyImage3D(Image3D_GPU_half* imageToCopy);


	void scalarImage(float scalar);// Multiply image by scalar
	void setImage(float value); // Set all pixel of image at value
	void addImage(Image3D_GPU_half* image2); // Add image 2 to current image
	void addImage(Image3D_GPU_half* image2, float lambda); // Add lambda*image2 to current image
	void positiveAddImage(Image3D_GPU_half* image2, float lambda); // Add lambda*image2 to current image with positiveness constraint
	void diffImage(Image3D_GPU_half* image2); // Sub image 2 to current image
	void diffImage(float lambda, Image3D_GPU_half* image2); // Sub image 2 to lambda*(current image)
	void multImage(Image3D_GPU_half* image2); // Multiply current image by image 2
	template <typename P> P scalarProductImage(Image3D_GPU_half* image2);// Scalar product between current image and image 2
	template <typename P> P getImageL1Norm();// Compute L1 Norm of image
	template <typename P> P getImageL2Norm();// Compute L2 Norm of image
	template <typename P> P getImageLpNorm(P p);// Compute Lp Norm of image
	template <typename P> P getImageHuberNorm(P threshold);// Compute Huber Norm of image
	template <typename P> P getImageMean();// Compute mean of image
	template <typename P> P getImageMeanSquare(); // Compute mean square of image
	template <typename P> P getImageStd();// Compute standard deviation of image

	void getImageSign(Image3D_GPU_half* signedImage);// Get signed image
	template <typename P> void getImageAbsPow(Image3D_GPU_half* absPowImage,P p);// Get absolute image raised to pth power

	void saveImage(string fileName); // Save image
	void saveMiddleSliceImage(string fileName); // Save middle slice image
	void loadImage(string fileName); // Load image
private:
	CUDAArchitecture* cudaArchitecture; // GPU architecture for sinogram

};




#endif /* IMAGE3D_HPP_ */
