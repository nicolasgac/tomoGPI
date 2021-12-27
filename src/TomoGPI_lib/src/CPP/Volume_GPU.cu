/*
 * Volume_GPU.cu
 *
 *      Author: gac
 */

#include "Volume_GPU.cuh"
#include "GPUConstant.cuh"
#include "kernel_segmentation.cuh"


template <typename T>
Volume_GPU<T>::Volume_GPU() : Volume<T>(){}

template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb){}


template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture,T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb), cudaArchitecture(cudaArchitecture){}

template <typename T>
Volume_GPU<T>::Volume_GPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_GPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture));
}

template <typename T>
Volume_GPU<T>::~Volume_GPU()
{
	delete this->getVolumeImage();
}

template <typename T>
Volume_GPU<T>::Volume_GPU(const Volume_GPU<T>& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_GPU<T>(*volumeToCopy.getVolumeImage()));
}

template <typename T>
Image3D_GPU<T>* Volume_GPU<T>::getVolumeImage() const
{
	return (Image3D_GPU<T>*)Volume<T>::getVolumeImage();
}

template <typename T>
void Volume_GPU<T>::setVolume(T value)
{
	this->getVolumeImage()->setImage(value);
}

template <typename T>
void Volume_GPU<T>::scalarVolume(T value)
{
	this->getVolumeImage()->scalarImage(value);
}

template <typename T>
void Volume_GPU<T>::addVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::addVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_GPU<T>::positiveAddVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_GPU<T>::diffVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::diffVolume(Volume_GPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage(), lambda);
}

template <typename T>
void Volume_GPU<T>::diffVolume(T lambda, Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::multVolume(Volume_GPU<T>* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

template <typename T>
T Volume_GPU<T>::scalarProductVolume(Volume_GPU<T>* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

template <typename T>
T Volume_GPU<T>::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

template <typename T>
T Volume_GPU<T>::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

template <typename T>
T Volume_GPU<T>::getVolumeLpNorm(T p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

template <typename T>
T Volume_GPU<T>::getVolumeHuberNorm(T p)
{
	return this->getVolumeImage()->getImageHuberNorm(p);
}

template <typename T>
T Volume_GPU<T>::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

template <typename T>
T Volume_GPU<T>::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

template <typename T>
T Volume_GPU<T>::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

template <typename T>
void Volume_GPU<T>::getVolumeSign(Volume_GPU<T>* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

template <typename T>
void Volume_GPU<T>::getVolumeAbsPow(Volume_GPU<T>* absPowVolume, T p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
template <typename T>
void Volume_GPU<T>::normalizeCenterVolumeMGI(Volume_GPU<T>* volume){

}

template <typename T>
double Volume_GPU<T>::sumNormalizedCenteredVolumeMGI(){

	return 0.0;

}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
template <typename T>
double Volume_GPU<T>::normalizeCenterClassesVolumeMGI(Volume_GPU<T>* volume){

	return 0.0;
}

// normalize volume (no centering) by the variances of another volume
template <typename T>
void Volume_GPU<T>::normalizeVolume(Volume_GPU<T>* volume){

}

template <typename T>
double Volume_GPU<T>::sumNormalizedVolume(){
	return 0.0;
}

// normalize volume
template <typename T>
void Volume_GPU<T>::normalizeVolume(){}

// center volume by the means of another volume
template <typename T>
void Volume_GPU<T>::centerVolumeMGI(Volume_GPU<T>* volume){}

// contours
template <typename T>
void Volume_GPU<T>::indicateContoursVolume(){}

// means for MGM
template <typename T>
void Volume_GPU<T>::computeMeansMGMVolume(Volume_GPU<T>* volume){}

//Gauss-Markov-Potts : joint MAP (MGI)
template <typename T>
void Volume_GPU<T>::maxMeansClassesMGI(double m0, double v0){}

template <typename T>
void Volume_GPU<T>::maxVariancesClassesMGI(double alpha0, double beta0){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGI_GPU(unsigned int numit, double tol){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGINoirs_GPU(){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGIBlancs_GPU(){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGI(unsigned int numit, double tol){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGIBlancs(){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGINoirs(){}

template <typename T>
double Volume_GPU<T>::computePottsEnergyMGI(){
	return 0;
}

//Gauss-Markov-Potts : joint MAP (MGM)
template <typename T>
void Volume_GPU<T>::maxMeansClassesMGM(double m0, double v0){}

template <typename T>
void Volume_GPU<T>::maxVariancesClassesMGM(double alpha0, double beta0){}

template <typename T>
void Volume_GPU<T>::maxMeansClassesMGMKnownContours(double m0, double v0){}

template <typename T>
void Volume_GPU<T>::maxVariancesClassesMGMKnownContours(double alpha0, double beta0){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGM(unsigned int numit, double tol){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGMBlancs(){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGMNoirs(){}

template <typename T>
double Volume_GPU<T>::computePottsEnergyMGM(){
	return 0;
}

template <typename T>
void Volume_GPU<T>::maxLabelsMGMFixedContours(unsigned int numit, double tol){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGMBlancsFixedContours(){}

template <typename T>
void Volume_GPU<T>::maxLabelsMGMNoirsFixedContours(){}

template <typename T>
double Volume_GPU<T>::computePottsEnergyMGMFixedContours(){
	return 0;
}

template <typename T>
void Volume_GPU<T>::selectContoursVolume(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::selectNoContoursVolume(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::centerMeanVolume(Volume_GPU<T>* volume){}

template <typename T>
double Volume_GPU<T>::normalizeCenterClassesVolumeMGM(Volume_GPU<T>* volume,Volume_GPU<T>* volume_mgm){
	return 0.0;
}

template <typename T>
double Volume_GPU<T>::sumNormalizedVolumeMGM(){
	return 0.0;
}

template <typename T>
void Volume_GPU<T>::centerMeanVolumeKnownContours(Volume_GPU<T>* volume){}

template <typename T>
double Volume_GPU<T>::normalizeCenterClassesVolumeMGMKnownContours(Volume_GPU<T>* volume,Volume_GPU<T>* volume_mgm){
	return 0.0;
}

template <typename T>
double Volume_GPU<T>::normalizeClassesDiffMeansVolume(Volume_GPU<T>* volume, Volume_GPU<T>* volume_means){
	return 0.0;
}

template <typename T>
double Volume_GPU<T>::sumNormalizedVolumeMGMKnownContours(){
	return 0.0;
}

template <typename T>
void Volume_GPU<T>::selectContoursVolume(){}

template <typename T>
void Volume_GPU<T>::selectNoContoursVolume(){}

template <typename T>
void Volume_GPU<T>::grad_xplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_xmoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_yplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_ymoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_zplus(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::grad_zmoins(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::sign_volume(Volume_GPU<T>* volume){}

template <typename T>
void Volume_GPU<T>::weightVolume(T* weights){}

template <typename T>
double Volume_GPU<T>::sumWeightedVolume(T* weights){
	return 0.0;
}


Volume_GPU_half::Volume_GPU_half() : Volume_GPU<half>(){}

Volume_GPU_half::Volume_GPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume_GPU<half>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb, cudaArchitecture){}

Volume_GPU_half::~Volume_GPU_half(){}

Volume_GPU_half::Volume_GPU_half(const Volume_GPU_half& volumeToCopy)
{
	this->setXVolumeSize(volumeToCopy.getXVolumeSize());
	this->setYVolumeSize(volumeToCopy.getYVolumeSize());
	this->setZVolumeSize(volumeToCopy.getZVolumeSize());

	this->setXVolumePixelNb(volumeToCopy.getXVolumePixelNb());
	this->setYVolumePixelNb(volumeToCopy.getYVolumePixelNb());
	this->setZVolumePixelNb(volumeToCopy.getZVolumePixelNb());

	this->setXVolumePixelSize(volumeToCopy.getXVolumePixelSize());
	this->setYVolumePixelSize(volumeToCopy.getYVolumePixelSize());
	this->setZVolumePixelSize(volumeToCopy.getZVolumePixelSize());

	this->setXVolumeCenterPixel(volumeToCopy.getXVolumeCenterPixel());
	this->setYVolumeCenterPixel(volumeToCopy.getYVolumeCenterPixel());
	this->setZVolumeCenterPixel(volumeToCopy.getZVolumeCenterPixel());

	this->setXVolumeStartPixel(volumeToCopy.getXVolumeStartPixel());
	this->setYVolumeStartPixel(volumeToCopy.getYVolumeStartPixel());
	this->setZVolumeStartPixel(volumeToCopy.getZVolumeStartPixel());

	this->setVolumeImage(new Image3D_GPU_half(*volumeToCopy.getVolumeImage()));
}

Image3D_GPU_half* Volume_GPU_half::getVolumeImage() const
{
	return (Image3D_GPU_half*)Volume_GPU<half>::getVolumeImage();
}

void Volume_GPU_half::setVolume(float value)
{
	this->getVolumeImage()->setImage(value);
}

void Volume_GPU_half::scalarVolume(float value)
{
	this->getVolumeImage()->scalarImage(value);
}

void Volume_GPU_half::addVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

void Volume_GPU_half::addVolume(Volume_GPU_half* volume2, float lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

void Volume_GPU_half::positiveAddVolume(Volume_GPU_half* volume2, float lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

void Volume_GPU_half::diffVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

void Volume_GPU_half::diffVolume(float lambda, Volume_GPU_half* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

void Volume_GPU_half::multVolume(Volume_GPU_half* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}
/*
template <>
float  Volume_GPU_half::scalarProductVolume(Volume_GPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage<float>(volume2->getVolumeImage());
}

template <>
float  Volume_GPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm<float>();
}

template <>
float  Volume_GPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm<float>();
}

template <>
float  Volume_GPU_half::getVolumeLpNorm(float p)
{
	return this->getVolumeImage()->getImageLpNorm<float>(p);
}

template <>
float  Volume_GPU_half::getVolumeHuberNorm(float threshold)
{
	return this->getVolumeImage()->getImageHuberNorm<float>(threshold);
}

template <>
float Volume_GPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean<float>();
}

template <>
float Volume_GPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare<float>();
}

template <>
float Volume_GPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd<float>();
}*/

//template <>
double Volume_GPU_half::scalarProductVolume(Volume_GPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage<double>(volume2->getVolumeImage());
}

//template <>
double Volume_GPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm<double>();
}

//template <>
double Volume_GPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm<double>();
}

//template <>
double Volume_GPU_half::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm<double>(p);
}

//template <>
double Volume_GPU_half::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm<double>(threshold);
}

//template <>
double Volume_GPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean<double>();
}

//template <>
double Volume_GPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare<double>();
}

//template <>
double Volume_GPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd<double>();
}

void Volume_GPU_half::getVolumeSign(Volume_GPU_half* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

//template <>
/*void Volume_GPU_half::getVolumeAbsPow(Volume_GPU_half* absPowVolume, float p)
{
	this->getVolumeImage()->getImageAbsPow<float>(absPowVolume->getVolumeImage(),p);
}*/

//template <>
void Volume_GPU_half::getVolumeAbsPow(Volume_GPU_half* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow<double>(absPowVolume->getVolumeImage(),p);
}

void Volume_GPU_half::saveVolume(string fileName)
{
	this->getVolumeImage()->saveImage(fileName);
}

void Volume_GPU_half::saveMiddleSliceVolume(string fileName)
{
	this->getVolumeImage()->saveMiddleSliceImage(fileName);
}

void Volume_GPU_half::loadVolume(string fileName)
{
	this->getVolumeImage()->loadImage(fileName);
}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
void Volume_GPU_half::normalizeCenterVolumeMGI(Volume_GPU_half* volume){

}

double Volume_GPU_half::sumNormalizedCenteredVolumeMGI(){

	return 0.0;

}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
double Volume_GPU_half::normalizeCenterClassesVolumeMGI(Volume_GPU_half* volume){
	return 0.0;
}

// normalize volume (no centering) by the variances of another volume
void Volume_GPU_half::normalizeVolume(Volume_GPU_half* volume){}

double Volume_GPU_half::sumNormalizedVolume(){
	return 0.0;
}

// normalize volume
void Volume_GPU_half::normalizeVolume(){}

// center volume by the means of another volume
void Volume_GPU_half::centerVolumeMGI(Volume_GPU_half* volume){}

// contours
void Volume_GPU_half::indicateContoursVolume(){}

// means for MGM
void Volume_GPU_half::computeMeansMGMVolume(Volume_GPU_half* volume){}

//Gauss-Markov-Potts : joint MAP (MGI)
void Volume_GPU_half::maxMeansClassesMGI(double m0, double v0){}

void Volume_GPU_half::maxVariancesClassesMGI(double alpha0, double beta0){}

void Volume_GPU_half::maxLabelsMGI(unsigned int numit, double tol){}

void Volume_GPU_half::maxLabelsMGIBlancs(){}

void Volume_GPU_half::maxLabelsMGINoirs(){}

double Volume_GPU_half::computePottsEnergyMGI(){
	return 0;
}

//Gauss-Markov-Potts : joint MAP (MGM)
void Volume_GPU_half::maxMeansClassesMGM(double m0, double v0){}

void Volume_GPU_half::maxVariancesClassesMGM(double alpha0, double beta0){}

void Volume_GPU_half::maxMeansClassesMGMKnownContours(double m0, double v0){}

void Volume_GPU_half::maxVariancesClassesMGMKnownContours(double alpha0, double beta0){}

void Volume_GPU_half::maxLabelsMGM(unsigned int numit, double tol){}

void Volume_GPU_half::maxLabelsMGMBlancs(){}

void Volume_GPU_half::maxLabelsMGMNoirs(){}

double Volume_GPU_half::computePottsEnergyMGM(){
	return 0;
}

void Volume_GPU_half::maxLabelsMGMFixedContours(unsigned int numit, double tol){}

void Volume_GPU_half::maxLabelsMGMBlancsFixedContours(){}

void Volume_GPU_half::maxLabelsMGMNoirsFixedContours(){}

double Volume_GPU_half::computePottsEnergyMGMFixedContours(){
	return 0;
}

void Volume_GPU_half::selectContoursVolume(){}

void Volume_GPU_half::selectNoContoursVolume(){}

void Volume_GPU_half::selectContoursVolume(Volume_GPU_half* volume){}

void Volume_GPU_half::selectNoContoursVolume(Volume_GPU_half* volume){}

void Volume_GPU_half::centerMeanVolume(Volume_GPU_half* volume){}

double Volume_GPU_half::normalizeCenterClassesVolumeMGM(Volume_GPU_half* volume,Volume_GPU_half* volume_mgm){
	return 0.0;
}

double Volume_GPU_half::sumNormalizedVolumeMGM(){
	return 0.0;
}

void Volume_GPU_half::centerMeanVolumeKnownContours(Volume_GPU_half* volume){}

double Volume_GPU_half::normalizeCenterClassesVolumeMGMKnownContours(Volume_GPU_half* volume,Volume_GPU_half* volume_mgm){
	return 0.0;
}

double Volume_GPU_half::normalizeClassesDiffMeansVolume(Volume_GPU_half* volume, Volume_GPU_half* volume_means){
	return 0.0;
}

double Volume_GPU_half::sumNormalizedVolumeMGMKnownContours(){
	return 0.0;
}

void Volume_GPU_half::grad_xplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_xmoins(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_yplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_ymoins(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_zplus(Volume_GPU_half* volume){}

void Volume_GPU_half::grad_zmoins(Volume_GPU_half* volume){}

void Volume_GPU_half::sign_volume(Volume_GPU_half* volume){}

#include "Volume_instances.cu"
#include "Volume_instances_GPU.cu"