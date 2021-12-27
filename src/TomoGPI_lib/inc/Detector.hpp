/*
 * Detector.hpp
 *
 *      Author: gac
 */

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_


class Detector{

public:

	Detector(float uDetectorSize, float vDetectorSize, unsigned long int uDetectorPixelNb, unsigned long int vDetectorPixelNb);
	Detector();
		~Detector();

	Detector & operator=(const Detector &detector);

	/* Physical Detector Parameters */
	float getUDetectorSize() const; // Get horizontal size of detector plane
	float getVDetectorSize() const; // Get vertical size of detector plane
	void setUDetectorSize(float uDetectorSize); // Set horizontal size of detector plane
	void setVDetectorSize(float vDetectorSize); // Set vertical size of detector plane

	float getUDetectorDecentering() const; // Get decentering in U
	float getVDetectorDecentering() const; // Get decentering in V
	void setUDetectorDecentering(float uDetectorDecentering); // Get decentering in U
	void setVDetectorDecentering(float vDetectorDecentering); // Get decentering in V

	float getDetectorTilte() const; // Get detector plane tilte
	void setDetectorTilte(float detectorTilte); // Set detector plane tilte

	/* Discrete Detector Parameters */
	unsigned long int getUDetectorPixelNb() const; // Get horizontal detector plane number of pixel
	unsigned long int getVDetectorPixelNb() const; // Get vertical detector plane number of pixel
	void setUDetectorPixelNb(unsigned long int uDetectorPixelNb); // Set horizontal detector plane number of pixel
	void setVDetectorPixelNb(unsigned long int vDetectorPixelNb); // Set vertical detector plane number of pixel

	float getUDetectorPixelSize() const; // Get size of detector plane pixel in U
	float getVDetectorPixelSize() const; // Get size of detector plane pixel in V
	void setUDetectorPixelSize(float uDetectorPixelSize); // Set size of detector plane pixel in U
	void setVDetectorPixelSize(float vDetectorPixelSize); // Set size of detector plane pixel in V

	float getUDetectorCenterPixel() const; // Get position of center pixel in U
	float getVDetectorCenterPixel() const; // Get position of center pixel in V
	void setUDetectorCenterPixel(float uCenterPixel); // Set position of center pixel in U
	void setVDetectorCenterPixel(float vCenterPixel); // Set position of center pixel in V

//#ifdef __CUDACC__
//	__host__ void  copyConstantGPU();
//#endif

private:
	/* Physical Detector Parameters */
	float uDetectorSize; // Horizontal size of detector plane
	float vDetectorSize; // Vertical size of detector plane

	float uDetectorDecentering; // U detector plane decentering
	float vDetectorDecentering; // V detector plane decentering

	float detectorTilte; // Detector plane tilte

	/* Discrete Detector Parameters */
	unsigned long int uDetectorPixelNb; // U detector plane number of pixel
	unsigned long int vDetectorPixelNb; // U detector plane number of pixel

	float uDetectorPixelSize; // U detector plane pixel size
	float vDetectorPixelSize; // V detector plane pixel size

	float uDetectorCenterPixel; // Position of detector center pixel in U
	float vDetectorCenterPixel; // Position of detector center pixel in V
};


#endif /* DETECTOR_HPP_ */
