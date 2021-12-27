/*
 * ConfigTiff.hpp
 *
 *      Author: gac
 */

#ifndef CONFIGTIFF_HPP_
#define CONFIGTIFF_HPP_

#include "Config.hpp"
#include "Sinogram3D_CPU.cuh"
#include "Image3D_CPU.cuh"

template<typename V> class ConfigTiff : public Config{

public:

	ConfigTiff();
	ConfigTiff(string configDirectoryName, string configFileName);
	~ConfigTiff();

	string getTiffName(); // Get name of tiff files
	unsigned short getTotalProjectionNb(); // Get total number of projections
	unsigned short getUsedProjectionNb(); // Get number of projection used
	unsigned short getUDetectorPixelInitialNb(); // Get initial U detector plane number of pixel
	unsigned short getVDetectorPixelInitialNb(); // Get initial V detector plane number of pixel
	unsigned short getUDetectorPixelFinalNb(); // Get final U detector plane number of pixel
	unsigned short getVDetectorPixelFinalNb(); // Get final V detector plane number of pixel
	unsigned short getUDetectorCenterFinal(); // Get final U detector center plane
	unsigned short getVDetectorCenterFinal(); // Get final V detector center plane
	unsigned short getProjectionSamplingRate(); // Get projection sampling rate
	double getNormalizationFactor(); // Get normalization factor
	double* getPhiValue(); // Get Value of angle of projections
	void setTiffName(string tiffName); // Set name of tiff files
	void setTotalProjectionNb(unsigned short totalProjectionNb); // Set total number of projections
	void setUsedProjectionNb(unsigned short usedProjectionNb); // Set number of projection used
	void setUDetectorPixelInitialNb(unsigned short uDetectorPixelInitialNb); // Set initial U detector plane number of pixel
	void setVDetectorPixelInitialNb(unsigned short vDetectorPixelInitialNb); // Set initial V detector plane number of pixel
	void setUDetectorPixelFinalNb(unsigned short uDetectorPixelFinalNb); // Set final U detector plane number of pixel
	void setVDetectorPixelFinalNb(unsigned short vDetectorPixelFinalNb); // Set final V detector plane number of pixel
	void setUDetectorCenterFinal(unsigned short uDetectorCenterFinal); // Set final U detector center plane
	void setVDetectorCenterFinal(unsigned short vDetectorCenterFinal); // Set final V detector center plane
	void setProjectionSamplingRate(unsigned short projectionSamplingRate); // Set projection sampling rate
	void setNormalizationFactor(double normalizationFactor); // Set normalization factor

	template <typename T> Sinogram3D_CPU<T> & createSinogram(); // Create sinogram from tiff files
	template <typename T> Sinogram3D_CPU<T> & createNormalizedMeanSinogram(); // Create sinogram from tiff files normalized by mean value of calibration image
	template <typename T> Sinogram3D_CPU<T> & createNormalizedLogSinogram();// create sinogram from tiff files normalized by pixels' values of calibration image (i.e. not mean value)
	template <typename T> Sinogram3D_CPU<T> & createNegativeLogSinogram(); // Create negative log sinogram (-log(sinogram)) from tiff files
	template <typename T> Sinogram3D_CPU<T> & createNormalizedMeanNegativeLogSinogram_5dig(); // Create negative log sinogram (-log(sinogram)) from tiff files normalized by mean value of calibration image, when tiff names have 5 digits
	template <typename T> Sinogram3D_CPU<T> & createNormalizedMeanNegativeLogSinogram_4dig(); // Create negative log sinogram (-log(sinogram)) from tiff files normalized by mean value of calibration image, when tiff names have 4 digits
	template <typename T> Sinogram3D_CPU<T> & createNormalizedNegativeLogSinogram(); // Create negative log sinogram (-log(sinogram)) from tiff files normalized by normalization factor specified in Tiff config file
	template <typename T> Sinogram3D_CPU<T> & createNonUniformNormalizedNegativeLogSinogram(); //// Create non uniform negative log sinogram (-log(sinogram)) from tiff files normalized by normalization factor specified in Tiff config file


private:
	/* Tiff files configuration*/
	string tiffName; // Name of tiff files
	unsigned short totalProjectionNb; // Total number of projection
	unsigned short usedProjectionNb; // Number of projection used
	unsigned short uDetectorPixelInitialNb; // Initial U detector plane number of pixel
	unsigned short vDetectorPixelInitialNb; // Initial V detector plane number of pixel
	unsigned short uDetectorPixelFinalNb; // Final U detector plane number of pixel
	unsigned short vDetectorPixelFinalNb; // Final V detector plane number of pixel
	unsigned short uDetectorCenterFinal; // Center U detector final plane
	unsigned short vDetectorCenterFinal; // Center V detector final plane
	unsigned short projectionSamplingRate; // Projection sampling rate (totalProjectionNb/usedProjectionNb)
	double normalizationFactor; // Normalization factor
	double* phiValue; // Value of angle of projections
};

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createSinogram()
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);
	sinogram->setSinogram(1.0);


	cout << "********** Uniform Projections **********" << endl;
	cout << "********** Start Creating sinogram **********" << endl;

	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);//5 : number of digits in tiff name
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=(T)tmpValue;
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}
	cout << "********** Sinogram created **********" << endl;

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNormalizedMeanSinogram()
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	unsigned int CalibrationEnergyNb;
	CalibrationEnergyNb=getConfigFileField<int>("CalibrationEnergyNb");
	ImageCalibration<unsigned short int> calibrationImage(getConfigFileField<int>("uCalibrationImagePixelNb"),getConfigFileField<int>("vCalibrationImagePixelNb"));
	calibrationImage.loadTiffImage(getConfigDirectoryName()+"/../Calibration/"+getConfigFileField<string>("calibrationTiffName"));
	float calibrationImageMeanValue = calibrationImage.getWhiteMeanValue(CalibrationEnergyNb);

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);
	sinogram->setSinogram(1.0);


	cout << "********** Uniform Projections **********" << endl;
	cout << "********** Start Creating sinogram **********" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=(T)tmpValue/calibrationImageMeanValue;
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}
	cout << "********** Sinogram created **********" << endl;

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNegativeLogSinogram()
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);

	cout << "********** Uniform Projections **********" << endl;
	cout << "********** Start Creating negative log sinogram **********" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue);
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}
	cout << "********** Sinogram created **********" << endl;

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNormalizedNegativeLogSinogram()
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);


	cout << "Creating negative log sinogram" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << this->getConfigDirectoryName()+"tif/"+tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/normalizationFactor);
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNormalizedMeanNegativeLogSinogram_5dig()//when tif name has 5 digits
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	unsigned int CalibrationEnergyNb;
	CalibrationEnergyNb=getConfigFileField<int>("CalibrationEnergyNb");
	ImageCalibration<unsigned short int> calibrationImage(getConfigFileField<int>("uCalibrationImagePixelNb"),getConfigFileField<int>("vCalibrationImagePixelNb"));

	calibrationImage.loadTiffImage(getConfigDirectoryName()+"/../Calibration/"+getConfigFileField<string>("calibrationTiffName"));

	float calibrationImageMeanValue = calibrationImage.getWhiteMeanValue(CalibrationEnergyNb);
	unsigned long int uNb_calib,vNb_calib;

	uNb_calib = calibrationImage.getXImagePixelNb();
	vNb_calib = (calibrationImage.getYImagePixelNb())/CalibrationEnergyNb;

	cout << "uNb_calib" << uNb_calib << endl;
	cout << "vNb_calib" << uNb_calib << endl;

	unsigned short uAddedNb_calib = uNb_calib-uDetectorPixelInitialNb;
	unsigned short vAddedNb_calib = vNb_calib-vDetectorPixelInitialNb;

	cout << "uAddedNb_calib" <<  uAddedNb_calib << endl;
	cout << "vAddedNb_calib" << vAddedNb_calib << endl;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);


	cout << "Creating negative log sinogram" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/calibrationImageMeanValue);
						//sinogram->getDataSinogram()[(u-uAddedNb/2)+(v-vAddedNb/2)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/(calibrationImage.getImageData()[(u-uAddedNb/2+uAddedNb_calib/2)+(v-vAddedNb/2+vAddedNb_calib/2)*uNb_calib+2*uNb_calib*vNb_calib]));
						//sinogram->getDataSinogram()[(u-uAddedNb/2)+(v-vAddedNb/2)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=calibrationImage.getImageData()[(u-uAddedNb/2+uAddedNb_calib/2)+(v-vAddedNb/2+vAddedNb_calib/2)*uNb_calib+2*uNb_calib*vNb_calib];
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNormalizedMeanNegativeLogSinogram_4dig()//when tif name has 4 digits
{
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	unsigned int CalibrationEnergyNb;
	CalibrationEnergyNb=getConfigFileField<int>("CalibrationEnergyNb");
	ImageCalibration<unsigned short int> calibrationImage(getConfigFileField<int>("uCalibrationImagePixelNb"),getConfigFileField<int>("vCalibrationImagePixelNb"));

	calibrationImage.loadTiffImage(getConfigDirectoryName()+"/../Calibration/"+getConfigFileField<string>("calibrationTiffName"));

	float calibrationImageMeanValue = calibrationImage.getWhiteMeanValue(CalibrationEnergyNb);
	unsigned long int uNb_calib,vNb_calib;

	uNb_calib = calibrationImage.getXImagePixelNb();
	vNb_calib = (calibrationImage.getYImagePixelNb())/CalibrationEnergyNb;

	cout << "uNb_calib" << uNb_calib << endl;
	cout << "vNb_calib" << uNb_calib << endl;

	unsigned short uAddedNb_calib = uNb_calib-uDetectorPixelInitialNb;
	unsigned short vAddedNb_calib = vNb_calib-vDetectorPixelInitialNb;

	cout << "uAddedNb_calib" <<  uAddedNb_calib << endl;
	cout << "vAddedNb_calib" << vAddedNb_calib << endl;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);


	cout << "Creating negative log sinogram" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%04d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/calibrationImageMeanValue);
						//sinogram->getDataSinogram()[(u-uAddedNb/2)+(v-vAddedNb/2)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/(calibrationImage.getImageData()[(u-uAddedNb/2+uAddedNb_calib/2)+(v-vAddedNb/2+vAddedNb_calib/2)*uNb_calib+2*uNb_calib*vNb_calib]));
						//sinogram->getDataSinogram()[(u-uAddedNb/2)+(v-vAddedNb/2)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=calibrationImage.getImageData()[(u-uAddedNb/2+uAddedNb_calib/2)+(v-vAddedNb/2+vAddedNb_calib/2)*uNb_calib+2*uNb_calib*vNb_calib];
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNormalizedLogSinogram()
{
	// Author : Camille Chapdelaine
	ifstream imageFile;
	V tmp[4];
	V tmpValue;
	int u,v,p;
	int phi = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	unsigned int CalibrationEnergyNb;
	CalibrationEnergyNb=getConfigFileField<int>("CalibrationEnergyNb");
	ImageCalibration<unsigned short int> calibrationImage(getConfigFileField<int>("uCalibrationImagePixelNb"),getConfigFileField<int>("vCalibrationImagePixelNb"));

	calibrationImage.loadTiffImage(getConfigDirectoryName()+"/../Calibration/"+getConfigFileField<string>("calibrationTiffName"));

	V* calibrationImage_data = calibrationImage.getImageData();
	unsigned long int uNb_calib,vNb_calib;

	uNb_calib = calibrationImage.getXImagePixelNb();
	vNb_calib = (calibrationImage.getYImagePixelNb())/CalibrationEnergyNb;

	cout << "uNb_calib" << uNb_calib << endl;
	cout << "vNb_calib" << uNb_calib << endl;

	unsigned short uAddedNb_calib = uNb_calib-uDetectorPixelInitialNb;
	unsigned short vAddedNb_calib = vNb_calib-vDetectorPixelInitialNb;

	cout << "uAddedNb_calib" <<  uAddedNb_calib << endl;
	cout << "vAddedNb_calib" << vAddedNb_calib << endl;

	char tiffNbTemp[255];
	Sinogram3D_CPU<T>* sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);


	cout << "Creating negative log sinogram" << endl;
	for(p = totalProjectionNb; p > (totalProjectionNb-projectionSamplingRate*usedProjectionNb);p-=projectionSamplingRate)
	{
		sprintf(tiffNbTemp,"_%05d.tif",p+1);
		imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
		if (imageFile.is_open())
		{
			cout << "Loading " << tiffName+tiffNbTemp << endl;
			imageFile.read ((char*)tmp, sizeof(V)*4);
			for(v=0;v<vDetectorPixelInitialNb;v++)
				for(u=0;u<uDetectorPixelInitialNb;u++)
				{
					imageFile.read ((char*)&tmpValue,sizeof(V));
					if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
					{
						sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/calibrationImage_data[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+(CalibrationEnergyNb-1)*calibrationImage.getYImagePixelNb()]);
					}
				}
		}
		else
		{
			cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
			cout << "Sinogram creation canceled" << endl;
			exit(EXIT_FAILURE);
		}
		phi++;
		cout << "Image " << tiffName+tiffNbTemp << " added to sinogram" << endl;
		imageFile.close();
	}

	return *sinogram;
}

template <typename V>
template <typename T> Sinogram3D_CPU<T> & ConfigTiff<V>::createNonUniformNormalizedNegativeLogSinogram()
{
	ifstream imageFile,pcpFile, iconfigProjectionFile, iconfigProjectionSchemeFile;
	ofstream oconfigProjectionFile;
	V tmp[4];
	V tmpValue;
	int u,v;
	int phi = 0;
	int phiSampling = 0;
	unsigned short inf_u = uDetectorCenterFinal-uDetectorPixelFinalNb/2;
	unsigned short inf_v = vDetectorCenterFinal-vDetectorPixelFinalNb/2;
	unsigned short sup_u = uDetectorCenterFinal+uDetectorPixelFinalNb/2;
	unsigned short sup_v = vDetectorCenterFinal+vDetectorPixelFinalNb/2;

	char tiffNbTemp[255];
	string tmp2;
	double phiValue=0.0;
	double phiNb=0.0;
	int area[2];
	int areaNb;
	int step;
	double *phiTab;
	int *phiTest;
	phiTab = (double *)malloc(totalProjectionNb*sizeof(double));
	phiTest = (int *)malloc(totalProjectionNb*sizeof(int));
	Sinogram3D_CPU<T>* sinogram;



	cout << "*************** Non uniform Projections ****************" << endl;
	cout << "Creating negative log sinogram" << endl;

	//	pcpFile.open((this->getConfigDirectoryName()+tiffName+"_.pcp").c_str(), ios::in|ios::binary);
	iconfigProjectionSchemeFile.open((this->getConfigDirectoryName()+"configProjectionScheme").c_str(), ios::in|ios::binary);
	oconfigProjectionFile.open((this->getConfigDirectoryName()+"configProjection").c_str(), ios::out|ios::binary);
	oconfigProjectionFile << "ImNb\t" << "RotPos" << endl;
	iconfigProjectionSchemeFile >> tmp2 >> areaNb;
	iconfigProjectionSchemeFile >> tmp2 >> tmp2 >> tmp2;
	if (oconfigProjectionFile.is_open() && iconfigProjectionSchemeFile.is_open())
	{
		for (int areaIdx = 0; areaIdx<areaNb; areaIdx++)
		{
			iconfigProjectionSchemeFile >> area[0] >> area[1] >> step;
			for(int phi = area[0];phi<=area[1];phi=phi+step)
			{
				phiTab[phi-1] = (phi-1)*360.0/totalProjectionNb;
				phiTest[phi-1] = 1;
			}
		}
		//		pcpFile >> tmp2 >> tmp2;
		for(int idx = 0;idx<=totalProjectionNb;idx++)
		{
			if(!(phiTest[idx]==0.0))
			{
				oconfigProjectionFile << idx+1 << "\t" << phiTab[idx] << endl;
				//				phiValue+= 360.0/totalProjectionNb;
			}
		}
		//	pcpFile.close();
		oconfigProjectionFile.close();

		cout << "*************** Selecting projections ****************" << endl;
		cout << "*************** Select projections in configProjection file and push enter ****************" << endl;
		getchar();
		iconfigProjectionFile.open((this->getConfigDirectoryName()+"configProjection").c_str(), ios::in|ios::binary);
		while(getline(iconfigProjectionFile, tmp2))
			phiNb++;

		cout << phiNb-2 << " projections selected" << endl;
		this->setUsedProjectionNb(phiNb-2);
		sinogram = new Sinogram3D_CPU<T>(this->uDetectorPixelFinalNb,this->vDetectorPixelFinalNb,this->usedProjectionNb,(CUDAArchitecture*)NULL);

		cout << "Please update usedProjectionNb parameter in configTiff file (usedProjectionNb " << phiNb-2 << ")" << endl;
		cout << "Please update projectionNb parameter in configCT file (projectionNb " << phiNb-2 << ")" << endl;
		cout << "Push enter when usedProjectionNb and projectionNb are updated" << endl;
		getchar();
		iconfigProjectionFile.clear();
		iconfigProjectionFile.seekg(0);
		stringstream ss;
		ss << phiNb-2;
#ifdef __linux__
		mkdir((this->getConfigDirectoryName()+"Projections_"+ ss.str()).c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
		mkdir((this->getConfigDirectoryName() + "Projections_" + ss.str()).c_str());
#endif
		this->phiValue = (double*)malloc((phiNb-2)*sizeof(double));

		cout << "*************** Creating Sinogram ****************" << endl;
		iconfigProjectionFile >> tmp2 >> tmp2;

		while((iconfigProjectionFile >> phi >> phiValue) && phiSampling < (phiNb-2))
		{
			this->phiValue[phiSampling] = (double)M_PI*(360.0-phiValue)/180.0;
			sprintf(tiffNbTemp,"_%05d.tif",phi);
			imageFile.open((this->getConfigDirectoryName()+"/../tif/"+tiffName+tiffNbTemp).c_str(), ios::in|ios::binary);
			if (imageFile.is_open())
			{
				cout << "Loading " << tiffName+tiffNbTemp << endl;
				imageFile.read ((char*)tmp, sizeof(V)*4);
				for(v=0;v<vDetectorPixelInitialNb;v++)
					for(u=0;u<uDetectorPixelInitialNb;u++)
					{
						imageFile.read ((char*)&tmpValue,sizeof(V));
						if(v >=inf_v && u >=inf_u && v <sup_v && u <sup_u)
						{
							sinogram->getDataSinogram()[(u-inf_u)+(v-inf_v)*uDetectorPixelFinalNb+phi*uDetectorPixelFinalNb*vDetectorPixelFinalNb]=-log((T)tmpValue/normalizationFactor);
						}
					}
				phiSampling++;
				imageFile.close();
			}
			else
			{
				cout << "Unable to open file " << tiffName+tiffNbTemp << endl;
				cout << "Non uniform Sinogram creation canceled " << endl;
				exit(EXIT_FAILURE);
			}
		}
		iconfigProjectionFile.close();
	}
	else
	{
		//		cout << "Unable to open file " << (this->getConfigDirectoryName()) << "/" << tiffName << "_.pcp" << endl;
		cout << "Unable to open file " << (this->getConfigDirectoryName()) << "configProjection" << endl;
		cout << "Unable to open file " << (this->getConfigDirectoryName()) << "configProjectionScheme" << endl;
		cout << "Non uniform Sinogram creation canceled " << endl;
		exit(EXIT_FAILURE);
	}

	return *sinogram;
}


#endif /* CONFIGTIFF_HPP_ */
