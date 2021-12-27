/*
 * Image3D_MGPU.hpp
 *
 *      Author: gac
 */

#ifndef IMAGE3D_MGPU_HPP_
#define IMAGE3D_MGPU_HPP_

#include "Image3D.cuh"


using namespace std;


class Acquisition;
template <typename T> class Sinogram3D;
template <typename T> class Volume;

template <typename T> class MGPUplan_Sino_s;


template <typename T> struct MGPUplan_Img_s
{
	int device;

    //Stream for asynchronous command execution
    cudaStream_t stream;

    struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	size_t SinoSize_d;
	T** Sinogram_d;

	size_t VolSize_d;
	T* Volume_d;
	//Added
	int z0_min;
	int z0_max;
	//! Le volume
	Volume<T>* volume_h;
	//! Le sinogramme
	Sinogram3D<T>* sinogram_h;
	// Computing Architecture
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
	//! Angle de initialisation (non utilise)
	int phi_start;
	int vn_start;
	int zn_start;
	int N_zn_par_solverthread;
	int nstreams;
	unsigned long long int size_sinogram;
	dim3 dimBlock;
	dim3 dimGrid;
	unsigned int SM;
	unsigned int numberCells;
	unsigned int N_vn_par_carte;
	unsigned int N_vn_par_solverthread;
	unsigned int N_vn_par_kernel;
	unsigned int N_ligne_par_carte;
	unsigned int N_vn_restant;
	Acquisition* acquisition;
	Detector* detector;

};




template<typename T>class Image3D_MGPU : public Image3D<T>{
public:

	Image3D_MGPU();
	Image3D_MGPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb,T* dataImage);
	Image3D_MGPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb, CUDAArchitecture* cudaArchitectureVolume);
	Image3D_MGPU(unsigned long int xImagePixelNb, unsigned long int yImagePixelNb, unsigned long int zImagePixelNb);
	~Image3D_MGPU();

	Image3D_MGPU(const Image3D_MGPU<T>& imageToCopy); // Copy-constructor

	CUDAArchitecture* getCUDAArchitecture() const;
	void setCUDAArchitecture(CUDAArchitecture* cudaArchitecture);

	void copyImage3D(Image3D_MGPU<T>* imageToCopy);
	void scalarImage(T scalar);// Multiply image by scalar
	void setImage(T value); // Set all pixel of image at value
	void addImage(Image3D_MGPU<T>* image2); // Add image 2 to current image
	void addImage(Image3D_MGPU<T>* image2, T lambda); // Add lambda*image2 to current image
	void positiveAddImage(Image3D_MGPU<T>* image2, T lambda); // Add lambda*image2 to current image with positiveness constraint
	void diffImage(Image3D_MGPU<T>* image2); // Sub image 2 to current image
	void diffImage(T lambda, Image3D_MGPU<T>* image2); // Sub image 2 to lambda*(current image)
	void diffImage(Image3D_MGPU<T>* image2, T lambda); // Sub lambda*(image 2) to current image
	void multImage(Image3D_MGPU<T>* image2); // Multiply current image by image 2
	T scalarProductImage(Image3D_MGPU<T>* image2);// Scalar product between current image and image 2
	T getImageL1Norm();// Compute L1 Norm of image
	T getImageL2Norm();// Compute L2 Norm of image
	T getImageLpNorm(T p);// Compute Lp Norm of image
	T getImageHuberNorm(T threshold);// Compute Huber Norm of image
	T getImageMean();// Compute mean of image
	T getImageMeanSquare(); // Compute mean square of image
	T getImageStd();// Compute standard deviation of image

	void getImageSign(Image3D_MGPU<T>* signedImage);// Get signed image
	void getImageAbsPow(Image3D_MGPU<T>* absPowImage,T p);// Get absolute image raised to pth power

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

/*
	 * MULTI GPU : MGPU
	 */
	 
	 void setMGPUplan_Img(MGPUplan_Img_s<T>* plan);
	 MGPUplan_Img_s<T>* getMGPUplan_Img();
	 
	 void setMGPUplan_Sino(MGPUplan_Sino_s<T>* plan);
	 MGPUplan_Sino_s<T>* getMGPUplan_Sino();
 
	 Volume_MGPU<T>* getVolume() const; // Get volume
	 void setVolume(Volume_MGPU<T>* volume); // Set volume

	 Sinogram3D_MGPU<T>* getSinogram3D() const; // Get volume
	 void setSinogram3D(Sinogram3D_MGPU<T>* sinogram); // Set volume


	 CUDAArchitecture* getCUDAArchitectureVolume() const;
	 void setCUDAArchitectureVolume(CUDAArchitecture* cudaArchitectureVolume); // Set CUDAProjectionArchitecture
	 CUDAProjectionArchitecture* getCUDAProjectionArchitecture() const;
	 void setCUDAProjectionArchitecture(CUDAProjectionArchitecture* cudaprojectionArchitecture); // Set CUDAProjectionArchitecture
	 Acquisition* getAcquisition() const; // Get Acquisition
	 Detector* getDetector() const; // Get detector
	 void setAcquisition(Acquisition* acquisition); // Set Acquisition
	 void setDetector(Detector* detector); // Set detector
 
 
	void loadImage(string fileName); // Load image
	void saveImage(string fileName); // Save image
	void saveImageIter(string fileName); // Save image

	void InitImage3D(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitImage3D_InitIter(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitImage3D_InitSG(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	void InitUpdate(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume1, Volume<T>* volume2, Sinogram3D<T>* sinogram);

	//void InitImgProj_v0(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);
	//void InitUpdate_Debug(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume1, Volume<T>* volume2, Sinogram3D<T>* sinogram);
	//Debug
	//void saveImageDebug(string fileName); // Save image
	//void InitImage3D_v1(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture* cudaprojectionArchitecture, Volume<T>* volume, Sinogram3D<T>* sinogram);

#ifdef __CUDACC__
	__host__ void copyConstantGPU();
#endif

private :
Volume<T>* volume;
Sinogram3D_MGPU<T>* sinogram;
MGPUplan_Img_s<T>* MGPUplan_Img;
	 MGPUplan_Sino_s<T>* MGPUplan_Sino;
Acquisition* acquisition;
Detector* detector;
CUDAArchitecture* cudaArchitecture; // GPU architecture for sinogram
	CUDAArchitecture* cudaArchitectureVolume; // GPU architecture for Volume
	CUDAProjectionArchitecture* cudaprojectionArchitecture;
};

#endif /* IMAGE3D_HPP_ */
