/*
 * Volume_CPU.cu
 *
 *      Author: gac
 */

#include "Volume_CPU.cuh"
#include "GPUConstant.cuh"
#include "kernel_segmentation.cuh"

/* Volume_CPU definition */
template <typename T>
Volume_CPU<T>::Volume_CPU() : Volume<T>(){}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,dataImage));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}


template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture, T* dataImage) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb,cudaArchitecture,dataImage));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::Volume_CPU(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb) : Volume<T>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setVolumeImage(new Image3D_CPU<T>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
	//Gauss-Markov-Potts : labels
	//this->setSegmentation(new Image3D_CPU<int>(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

template <typename T>
Volume_CPU<T>::~Volume_CPU()
{
	delete this->getVolumeImage();
}

/*template <typename T>
Volume_CPU<T>&  Volume_CPU<T>::operator=(const Volume_CPU<T> &volumeToCopy)
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

		this->getVolumeImage()->copyImage3D(volumeToCopy.getVolumeImage());


	return *this;
}*/

template <typename T>
Volume_CPU<T>::Volume_CPU(const Volume_CPU<T>& volumeToCopy)
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

	this->setVolumeImage(new Image3D_CPU<T>(*volumeToCopy.getVolumeImage()));
}

template <typename T>
Image3D_CPU<T>* Volume_CPU<T>::getVolumeImage() const
{
	return (Image3D_CPU<T>*)Volume<T>::getVolumeImage();
}

template <typename T>
void Volume_CPU<T>::setVolume(T value)
{
	this->getVolumeImage()->setImage(value);
}

template <typename T>
void Volume_CPU<T>::scalarVolume(T value)
{
	this->getVolumeImage()->scalarImage(value);
}

template <typename T>
void Volume_CPU<T>::addVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::addVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_CPU<T>::positiveAddVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

template <typename T>
void Volume_CPU<T>::diffVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::diffVolume(T lambda, Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::diffVolume(Volume_CPU<T>* volume2, T lambda)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage(), lambda);
}

template <typename T>
void Volume_CPU<T>::multVolume(Volume_CPU<T>* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

template <typename T>
double Volume_CPU<T>::scalarProductVolume(Volume_CPU<T>* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

template <typename T>
double Volume_CPU<T>::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

template <typename T>
double Volume_CPU<T>::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

template <typename T>
double Volume_CPU<T>::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

template <typename T>
double Volume_CPU<T>::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm(threshold);
}

template <typename T>
double Volume_CPU<T>::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

template <typename T>
double Volume_CPU<T>::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

template <typename T>
double Volume_CPU<T>::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

template <typename T>
void Volume_CPU<T>::getVolumeSign(Volume_CPU<T>* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

template <typename T>
void Volume_CPU<T>::getVolumeAbsPow(Volume_CPU<T>* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}


// center each pixel and normalize them (MGI) by the means and the variances of another volume
template <typename T>
void Volume_CPU<T>::normalizeCenterVolumeMGI(Volume_CPU<T>* volume){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=volume->getVolumeData();
	int* labelsVolume=volume->getLabels();
	double* meansVolume=volume->getMeansClasses();
	double* variancesVolume=volume->getVariancesClasses();
	T* data=this->getVolumeData();
	int k=0;//label

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				data[adresse_voxel]=(volumeData[adresse_voxel]-T(meansVolume[k]))/T(variancesVolume[k]);
			}
		}
	}

}

template <typename T>
double Volume_CPU<T>::sumNormalizedCenteredVolumeMGI(){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=this->getVolumeData();
	int* labelsVolume=this->getLabels();
	double* meansVolume=this->getMeansClasses();
	double* variancesVolume=this->getVariancesClasses();
	int k=0;//label
	double sum_normalized_center=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				sum_normalized_center+=(double(volumeData[adresse_voxel])-meansVolume[k])*(double(volumeData[adresse_voxel])-meansVolume[k])/variancesVolume[k];
			}
		}
	}

	return sum_normalized_center;

}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
template <typename T>
double Volume_CPU<T>::normalizeCenterClassesVolumeMGI(Volume_CPU<T>* volume){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=volume->getVolumeData();
	int* labelsVolume=volume->getLabels();
	double* meansVolume=volume->getMeansClasses();
	double* variancesVolume=volume->getVariancesClasses();
	T* data=this->getVolumeData();
	int k=0;//label
	double sum_normalized_center=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				data[adresse_voxel]=(volumeData[adresse_voxel]-T(meansVolume[k]))/T(variancesVolume[k]);
				sum_normalized_center+=(double(volumeData[adresse_voxel])-meansVolume[k])*(double(volumeData[adresse_voxel])-meansVolume[k])/variancesVolume[k];
			}
		}
	}

	return sum_normalized_center;

}

// normalize volume (no centering) by the variances of another volume
template <typename T>
void Volume_CPU<T>::normalizeVolume(Volume_CPU<T>* volume){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=volume->getVolumeData();
	int* labelsVolume=volume->getLabels();
	double* variancesVolume=volume->getVariancesClasses();
	T* data=this->getVolumeData();
	int k=0;//label

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				data[adresse_voxel]=volumeData[adresse_voxel]/T(variancesVolume[k]);
			}
		}
	}

}

template <typename T>
double Volume_CPU<T>::sumNormalizedVolume(){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=this->getVolumeData();
	int* labelsVolume=this->getLabels();
	double* variancesVolume=this->getVariancesClasses();
	int k=0;//label
	double sum_normalized=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				sum_normalized+=double(volumeData[adresse_voxel])*double(volumeData[adresse_voxel])/variancesVolume[k];
			}
		}
	}

	return sum_normalized;

}

// normalize volume
template <typename T>
void Volume_CPU<T>::normalizeVolume(){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int* labelsVolume=this->getLabels();
	double* variancesVolume=this->getVariancesClasses();
	T* data=this->getVolumeData();
	int k=0;//label

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				data[adresse_voxel]=data[adresse_voxel]/T(variancesVolume[k]);
			}
		}
	}

}

// center volume by the means of another volume
template <typename T>
void Volume_CPU<T>::centerVolumeMGI(Volume_CPU<T>* volume){

	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	T* volumeData=volume->getVolumeData();
	int* labelsVolume=volume->getLabels();
	double* meansVolume=volume->getMeansClasses();
	T* data=this->getVolumeData();
	int k=0;//label

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labelsVolume[adresse_voxel];
				data[adresse_voxel]=volumeData[adresse_voxel]-T(meansVolume[k]);
			}
		}
	}

}

//Gauss-Markov-Potts : joint MAP (MGI)
template <typename T>
void Volume_CPU<T>::maxMeansClassesMGI(double m0, double v0){

	std::cout << "\tMAP (MGI) for means of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	int k;

	// initialization
	double* m_classes=(double*) malloc(K*sizeof(double));//means of the classes
	unsigned long int* numVoxClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels in each class
	for(k=0;k<K;k++){
		m_classes[k]=0;
		numVoxClasses[k]=0;
	}

	// means
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				numVoxClasses[k]=numVoxClasses[k]+1;
				m_classes[k]+=(double) data_volume[adresse_voxel];
			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		m_classes[k]=(m0/v0)+(m_classes[k]/v_classes[k]);
		m_classes[k]=m_classes[k]/((1/v0)+(((double) numVoxClasses[k])/v_classes[k]));
	}

	// set
	this->setMeansClasses(m_classes);
	free(numVoxClasses);
}

template <typename T>
void Volume_CPU<T>::maxVariancesClassesMGI(double alpha0, double beta0){

	std::cout << "\tMAP (MGI) for variances of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	int K=this->getNumberClasses();//number of classes
	int k;

	// initialization
	double* v_classes=(double*) malloc(K*sizeof(double));//variances of the classes
	unsigned long int* numVoxClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels in each class
	for(k=0;k<K;k++){
		v_classes[k]=0;
		numVoxClasses[k]=0;
	}

	// variances
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				numVoxClasses[k]=numVoxClasses[k]+1;
				v_classes[k]+=((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]));
			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		v_classes[k]=beta0+0.5*v_classes[k];
		v_classes[k]=v_classes[k]/(alpha0+0.5*double(numVoxClasses[k])+1);
	}

	// set
	this->setVariancesClasses(v_classes);
	free(numVoxClasses);

}


template <typename T>
void Volume_CPU<T>::maxLabelsMGI(unsigned int numit, double tol){

	std::cout << "\tSegmentation MGI, all on CPU" << std::endl;
	// numit : maximum number of iterations
	// tol : tolerance to stop the algorithm
	double potts_energy=0;
	double* potts_energy_vect=(double*) malloc(numit*sizeof(double));

	unsigned int it=0;//iteration
	double diff_potts=10*tol;//to stop the algorithm
	/*std::cout << "\tGamma "<<std::to_string(this->getGammaPotts()) << std::endl;
	std::cout << "\tNumit "<<std::to_string(numit) << std::endl;
	std::cout << "\tTol "<<std::to_string(tol) << std::endl;
	std::cout << "\tK "<<std::to_string(this->getNumberClasses()) << std::endl;*/
	//this->getSegmentation()->saveMiddleSliceImage("segmentation_init.v");

	while(it<numit && diff_potts>tol){

		//blancs


		this->maxLabelsMGIBlancs();

		//noirs
		this->maxLabelsMGINoirs();

		//Potts energy
		potts_energy=this->computePottsEnergyMGI();
		potts_energy_vect[it]=potts_energy;

		// update
		if(it>0){
			diff_potts=abs(potts_energy_vect[it]-potts_energy_vect[it-1])/abs(potts_energy_vect[it-1]);
		}
		std::cout << "\tIteration "<<std::to_string(it+1)<<" of segmentation MGI completed, all on CPU" << std::endl;
		it+=1;

	}
	//this->getSegmentation()->saveMiddleSliceImage("segmentation_finale.v");
}


template <typename T>
void Volume_CPU<T>::maxLabelsMGI_GPU(unsigned int numit, double tol){
		std::cout << "\tSegmentation MGI, all on GPU Nicolas" << std::endl;
		// numit : maximum number of iterations
		// tol : tolerance to stop the algorithm

		//dimension du volume
		unsigned long int Nx=this->getXVolumePixelNb();
		unsigned long int Ny=this->getYVolumePixelNb();
		unsigned long int Nz=this->getZVolumePixelNb();

		//Taille Volume
		unsigned long long int size_vol= Nx*Ny*Nz;

		//
		size_t size_vol_o, size_label_o;

		size_label_o = size_vol*sizeof(int);
		size_vol_o = size_vol*sizeof(T);

		double potts_energy=0;
		double* potts_energy_vect=(double*) malloc(numit*sizeof(double));

		unsigned int it=0;//iteration
		double diff_potts=10*tol;//to stop the algorithm

		T* h_data_volume = this->getVolumeData();//data
		int* h_labels = this->getLabels();//labels

		T* d_data_volume;
		int* d_labels = (int*)malloc(size_label_o);

		checkCudaErrors( cudaMalloc( (void**) &d_labels , size_label_o ));
		checkCudaErrors( cudaMalloc( (void**) &d_data_volume , size_vol_o ));

		checkCudaErrors( cudaMemcpy( d_labels , h_labels, size_label_o , cudaMemcpyHostToDevice));
		checkCudaErrors( cudaMemcpy( d_data_volume, h_data_volume, size_vol_o ,cudaMemcpyHostToDevice));


		//Copie des données dans la mémoire constante
		this->copyConstantGPU();

		//Si pas de copie sur la mémoire de constante
		//int Kclasse = this->getNumberClasses();
		//double gammaPotts = this->getGammaPotts();
		//double* h_meanclasses = this->getMeansClasses();
		//double* h_varianceclasses = this->getVariancesClasses();
		//double* h_energySingleton = this->getEnergySingleton();

		//size_t size_double;
		//size_double = Kclasse*sizeof(double);

		//double *d_meanclasses;
		//double *d_varianceclasses;
		//double *d_energySingleton;

		checkCudaErrors(cudaSetDevice(0));

		//checkCudaErrors( cudaMalloc( (void**) &d_meanclasses , size_double ));
		//checkCudaErrors( cudaMalloc( (void**) &d_varianceclasses , size_double ));
		//checkCudaErrors( cudaMalloc( (void**) &d_energySingleton , size_double ));

		//checkCudaErrors( cudaMemcpy( d_meanclasses, h_meanclasses, size_double ,cudaMemcpyHostToDevice));
		//checkCudaErrors( cudaMemcpy( d_varianceclasses, h_varianceclasses, size_double ,cudaMemcpyHostToDevice));
		//checkCudaErrors( cudaMemcpy( d_energySingleton, h_energySingleton, size_double ,cudaMemcpyHostToDevice));

		//fin des copies

		//dimension de la grille pour un volume de 256
		dim3 numberThreadPerBlock(16,16,1);
		dim3 numberBlockPerGrid(16,16,256);

		while(it<numit && diff_potts>tol){

			//blancs
			kernel_maxLabelsMGINoirs<<<numberBlockPerGrid,numberThreadPerBlock>>>(d_data_volume, d_labels);
			//Actualisation label
			//noirs
			kernel_maxLabelsMGIBlancs<<<numberBlockPerGrid,numberThreadPerBlock>>>(d_data_volume, d_labels);


			/*
			std::cout << "Lancement maxLabelsMGIBlancs_GPU" << endl;
			kernel_maxLabelsMGIBlancs<<<numberBlockPerGrid,numberThreadPerBlock>>>(d_data_volume, d_labels,  Kclasse,  Nx,  Ny,  Ny,  gammaPotts, d_meanclasses, d_varianceclasses, d_energySingleton);
			std::cout << "Lancement maxLabelsMGINoirs_GPU" << endl;
			kernel_maxLabelsMGINoirs<<<numberBlockPerGrid,numberThreadPerBlock>>>(d_data_volume, d_labels,  Kclasse,  Nx,  Ny,  Ny,  gammaPotts, d_meanclasses, d_varianceclasses, d_energySingleton);
			*/

			checkCudaErrors(cudaMemcpy( h_labels , d_labels , size_label_o, cudaMemcpyDeviceToHost ));

			this->setLabels(h_labels);

			//Potts energy
			potts_energy=this->computePottsEnergyMGI();
			potts_energy_vect[it]=potts_energy;

			// update
			if(it>0){
				diff_potts=abs(potts_energy_vect[it]-potts_energy_vect[it-1])/abs(potts_energy_vect[it-1]);
			}
			std::cout << " diff_potts : " << diff_potts << std::endl;
			std::cout << "\tIteration "<<std::to_string(it+1)<<" of segmentation MGI completed, all on GPU" << std::endl;
			it+=1;
			std::cout << "\tSegmentation MGI, all on GPU Nicolas" << std::endl;
		}

		checkCudaErrors( cudaFree( d_data_volume ));
		checkCudaErrors( cudaFree( d_labels ));

		//si pas de copie sur la mémoire constante
		//checkCudaErrors( cudaFree( d_meanclasses ));
		//checkCudaErrors( cudaFree( d_varianceclasses ));
		//checkCudaErrors( cudaFree( d_energySingleton ));

}


template <typename T>
void Volume_CPU<T>::maxLabelsMGINoirs_GPU(T* d_data_volume, int* d_labels, int Kclasse, unsigned long  Nx, unsigned long Ny, unsigned long  Nz, int gammaPotts, double* d_meanclasses, double* d_varianceclasses, double* d_energySingleton){

	dim3 numberThreadPerBlock(16,16,16);
	dim3 numberBlockPerGrid(16,16,16);
	std::cout << "Lancement maxLabelsMGINoirs_GPU" << endl;
	/*
	dim3 dimBlock(this->getCUDAProjectionArchitecture()->getXThreadNb(),this->getCUDAProjectionArchitecture()->getYThreadNb(),this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	dim3 dimGrid(estimatedSinogram->getUSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getXThreadNb(),estimatedSinogram->getVSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getYThreadNb(),estimatedSinogram->getProjectionSinogramNb()/this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	 */

	//kernel_maxLabelsMGINoirs<<<numberBlockPerGrid,numberThreadPerBlock>>>(d_data_volume, d_labels,  Kclasse,  Nx,  Ny,  Ny,  gammaPotts, d_meanclasses, d_varianceclasses, d_energySingleton);
}

template <typename T>
void Volume_CPU<T>::maxLabelsMGIBlancs_GPU(T* d_data_volume, int* d_labels, int Kclasse, unsigned long  Nx, unsigned long Ny, unsigned long  Nz, int gammaPotts, double* d_meanclasses, double* d_varianceclasses, double* d_energySingleton){


	dim3 numberThreadPerBlock(16,16,16);
	dim3 numberBlockPerGrid(16,16,16);

	std::cout << "Lancement maxLabelsMGIBlancs_GPU" << endl;
	/*
	dim3 dimBlock(this->getCUDAProjectionArchitecture()->getXThreadNb(),this->getCUDAProjectionArchitecture()->getYThreadNb(),this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	dim3 dimGrid(estimatedSinogram->getUSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getXThreadNb(),estimatedSinogram->getVSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getYThreadNb(),estimatedSinogram->getProjectionSinogramNb()/this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	*/
	//kernel_maxLabelsMGIBlancs <<< numberBlockPerGrid, numberThreadPerBlock >>>(d_data_volume, d_labels,  Kclasse,  Nx,  Ny,  Nz,  gammaPotts, d_meanclasses, d_varianceclasses, d_energySingleton);
}

template <typename T>
void Volume_CPU<T>::maxLabelsMGIBlancs(){

	//std::cout << "\tSegmentation MGI for group blancs, all on CPU" << std::endl;
	// update only labels of voxels in group "blancs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();

	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_blancs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels

	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient

	double* proba_class=(double*) malloc(K*sizeof(double));
	double likelihood=0;
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				if(in_blancs){
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					//(-1,0,0)
					if(xn>0){
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//maximum of probability
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-m_classes[0])*(double(data_volume[adresse_voxel])-m_classes[0]))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];

					for(k=1;k<K;k++){
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
					labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);

}


template <typename T>
void Volume_CPU<T>::maxLabelsMGINoirs(){

	//std::cout << "\tSegmentation MGI for group noirs, all on CPU" << std::endl;
	// update only labels of voxels in group "noirs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_noirs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels

	int K=this->getNumberClasses();//number of classes
	double gamma_potts=this->getGammaPotts();//Potts coefficient

	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton


	double* proba_class= (double*) malloc(K*sizeof(double));
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	double likelihood=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_noirs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_noirs=in_noirs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				in_noirs=!in_noirs;
				if(in_noirs){
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					//(-1,0,0)
					if(xn>0){
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//maximum of probability
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-m_classes[0])*(double(data_volume[adresse_voxel])-m_classes[0]))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];
					for(k=1;k<K;k++){
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
					labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);

}





template <typename T>
double Volume_CPU<T>::computePottsEnergyMGI(){

	std::cout << "\tPotts energy MGI, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;//label
	//compare labels
	unsigned int deltaz_moins,deltaz_plus;
	unsigned int deltay_moins,deltay_plus;
	unsigned int deltax_moins,deltax_plus;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient

	double energy_neighbours=0;
	double energy_data=0;

	// initialize
	double potts_energy=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];

				//(-1,0,0)
				deltax_moins=0;
				if(xn>0){
					if(k==labels[(xn-1)+yn*Nx+zn*Nx*Ny]){
						deltax_moins=1;
					}
				}
				//(1,0,0)
				deltax_plus=0;
				if(xn<Nx-1){
					if(k==labels[(xn+1)+yn*Nx+zn*Nx*Ny]){
						deltax_plus=1;
					}
				}
				//(0,-1,0)
				deltay_moins=0;
				if(yn>0){
					if(k==labels[xn+(yn-1)*Nx+zn*Nx*Ny]){
						deltay_moins=1;
					}
				}
				//(0,1,0)
				deltay_plus=0;
				if(yn<Ny-1){
					if(k==labels[xn+(yn+1)*Nx+zn*Nx*Ny]){
						deltay_plus=1;
					}
				}
				//(0,0,-1)
				deltaz_moins=0;
				if(zn>0){
					if(k==labels[xn+yn*Nx+(zn-1)*Nx*Ny]){
						deltaz_moins=1;
					}
				}
				//(0,0,1)
				deltaz_plus=0;
				if(zn<Nz-1){
					if(k==labels[xn+yn*Nx+(zn+1)*Nx*Ny]){
						deltaz_plus=1;
					}
				}

				energy_neighbours=gamma_potts*double(deltax_moins+deltax_plus+deltay_moins+deltay_plus+deltaz_moins+deltaz_plus);
				energy_data=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]))/v_classes[k]);

				potts_energy+=(energy_data+energy_neighbours);
			}
		}
	}

	return potts_energy;
}

//Gauss-Markov-Potts : joint MAP (MGM)
template <typename T>
void Volume_CPU<T>::maxMeansClassesMGM(double m0, double v0){

	std::cout << "\tMAP (MGM) for means of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	bool contour;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	int k;

	// initialization
	double* m_classes=(double*) malloc(K*sizeof(double));//means of the classes
	unsigned long int* numVoxContoursClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels on contours in each class
	for(k=0;k<K;k++){
		m_classes[k]=0;
		numVoxContoursClasses[k]=0;
	}

	// means
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				if(contour){
					numVoxContoursClasses[k]=numVoxContoursClasses[k]+1;
					m_classes[k]+=(double) data_volume[adresse_voxel];
				}

			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		m_classes[k]=(m0/v0)+(m_classes[k]/v_classes[k]);
		m_classes[k]=m_classes[k]/((1/v0)+(((double) numVoxContoursClasses[k])/v_classes[k]));
	}

	// set
	this->setMeansClasses(m_classes);
	free(numVoxContoursClasses);

}

//Gauss-Markov-Potts : joint MAP (MGM)
template <typename T>
void Volume_CPU<T>::maxMeansClassesMGMKnownContours(double m0, double v0){

	std::cout << "\tMAP (MGM) for means of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	bool contour;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	bool* contours_volume=this->getBooleanContours();// contours
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	int k;

	// initialization
	double* m_classes=(double*) malloc(K*sizeof(double));//means of the classes
	unsigned long int* numVoxContoursClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels on contours in each class
	for(k=0;k<K;k++){
		m_classes[k]=0;
		numVoxContoursClasses[k]=0;
	}

	// means
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contours_volume[adresse_voxel];
				if(contour){
					numVoxContoursClasses[k]=numVoxContoursClasses[k]+1;
					m_classes[k]+=(double) data_volume[adresse_voxel];
				}

			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		m_classes[k]=(m0/v0)+(m_classes[k]/v_classes[k]);
		m_classes[k]=m_classes[k]/((1/v0)+(((double) numVoxContoursClasses[k])/v_classes[k]));
	}

	// set
	this->setMeansClasses(m_classes);
	free(numVoxContoursClasses);

}

template <typename T>
void Volume_CPU<T>::maxVariancesClassesMGM(double alpha0, double beta0){

	std::cout << "\tMAP (MGM) for variances of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	int K=this->getNumberClasses();//number of classes
	int k;
	float mean_neighbours;//mean of the neighbours
	double mean_voxel;
	int number_neighbours;
	bool contour;// contour

	// initialization
	double* v_classes=(double*) malloc(K*sizeof(double));//variances of the classes
	unsigned long int* numVoxClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels in each class
	for(k=0;k<K;k++){
		v_classes[k]=0;
		numVoxClasses[k]=0;
	}

	// variances
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				numVoxClasses[k]=numVoxClasses[k]+1;
				contour=false;
				mean_neighbours=0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;
				mean_voxel=double(mean_neighbours);

				if(contour){
					v_classes[k]+=((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]));
				}else{
					v_classes[k]+=((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel));
				}

			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		v_classes[k]=beta0+0.5*v_classes[k];
		v_classes[k]=v_classes[k]/(alpha0+0.5*double(numVoxClasses[k])+1);
	}

	// set
	this->setVariancesClasses(v_classes);
	free(numVoxClasses);

}

template <typename T>
void Volume_CPU<T>::maxVariancesClassesMGMKnownContours(double alpha0, double beta0){

	std::cout << "\tMAP (MGM) for variances of the classes, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	bool* contours_volume=this->getBooleanContours();// contours
	double* m_classes=this->getMeansClasses();//means of the classes
	int K=this->getNumberClasses();//number of classes
	int k;
	float mean_neighbours;//mean of the neighbours
	double mean_voxel;
	int number_neighbours;
	bool contour;// contour

	// initialization
	double* v_classes=(double*) malloc(K*sizeof(double));//variances of the classes
	unsigned long int* numVoxClasses=(unsigned long int*) malloc(K*sizeof(unsigned long int));// number of voxels in each class
	for(k=0;k<K;k++){
		v_classes[k]=0;
		numVoxClasses[k]=0;
	}

	// variances
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				numVoxClasses[k]=numVoxClasses[k]+1;
				contour=contours_volume[adresse_voxel];
				mean_neighbours=0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;
				mean_voxel=double(mean_neighbours);

				if(contour){
					v_classes[k]+=((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k]));
				}else{
					v_classes[k]+=((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel));
				}

			}
		}
	}

	//prior
	for(k=0;k<K;k++){
		v_classes[k]=beta0+0.5*v_classes[k];
		v_classes[k]=v_classes[k]/(alpha0+0.5*double(numVoxClasses[k])+1);
	}

	// set
	this->setVariancesClasses(v_classes);
	free(numVoxClasses);

}

template <typename T>
void Volume_CPU<T>::computeMeansMGMVolume(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;//label

	T* volume_means_mgm=this->getVolumeData();//means MGM
	T* data_volume=volume->getVolumeData();//data
	int* labels=volume->getLabels();//labels
	double* m_classes=volume->getMeansClasses();//means of the classes
	bool* contour_volume=volume->getBooleanContours();// contour
	bool contour;//contour
	double mean_voxel;//mean for the voxel
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contour_volume[adresse_voxel];
				mean_neighbours=0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}

				// mean
				mean_neighbours=mean_neighbours/number_neighbours;
				if(contour){
					mean_voxel=m_classes[k];
				}else{
					mean_voxel=double(mean_neighbours);
				}

				volume_means_mgm[adresse_voxel]=T(mean_voxel);
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::computeMeanNeighbours(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* volume_mean_neighbours=this->getVolumeData();//means  of neighbours
	T* data_volume=volume->getVolumeData();//data
	float mean_neighbours;
	int number_neighbours;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				mean_neighbours=0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}

				// mean
				mean_neighbours=mean_neighbours/number_neighbours;
				volume_mean_neighbours[adresse_voxel]=T(mean_neighbours);

			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::indicateContoursVolume(){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	bool* contours_volume=this->getBooleanContours();

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}

				contours_volume[adresse_voxel]=contour;
			}
		}
	}


}

template <typename T>
void Volume_CPU<T>::maxLabelsMGM(unsigned int numit, double tol){

	std::cout << "\tSegmentation MGM, all on CPU" << std::endl;
	// numit : maximum number of iterations
	// tol : tolerance to stop the algorithm
	double potts_energy=0;
	double* potts_energy_vect=(double*) malloc(numit*sizeof(double));

	unsigned int it=0;//iteration
	double diff_potts=10*tol;//to stop the algorithm
	while(it<numit && diff_potts>tol){

		//blancs
		this->maxLabelsMGMBlancs();

		//noirs
		this->maxLabelsMGMNoirs();

		//Potts energy
		potts_energy=this->computePottsEnergyMGM();
		potts_energy_vect[it]=potts_energy;

		// update
		if(it>0){
			diff_potts=abs(potts_energy_vect[it]-potts_energy_vect[it-1])/abs(potts_energy_vect[it-1]);
		}
		std::cout << "\tIteration "<<std::to_string(it+1)<<" of segmentation MGM completed, all on CPU" << std::endl;
		it+=1;

	}

}

template <typename T>
void Volume_CPU<T>::maxLabelsMGMFixedContours(unsigned int numit, double tol){

	std::cout << "\tSegmentation MGM, all on CPU" << std::endl;
	// numit : maximum number of iterations
	// tol : tolerance to stop the algorithm
	double potts_energy=0;
	double* potts_energy_vect=(double*) malloc(numit*sizeof(double));

	unsigned int it=0;//iteration
	double diff_potts=10*tol;//to stop the algorithm
	while(it<numit && diff_potts>tol){

		//blancs
		this->maxLabelsMGMBlancsFixedContours();

		//noirs
		this->maxLabelsMGMNoirsFixedContours();

		//Potts energy
		potts_energy=this->computePottsEnergyMGMFixedContours();
		potts_energy_vect[it]=potts_energy;

		// update
		if(it>0){
			diff_potts=abs(potts_energy_vect[it]-potts_energy_vect[it-1])/abs(potts_energy_vect[it-1]);
		}
		std::cout << "\tIteration "<<std::to_string(it+1)<<" of segmentation MGM with fixed contours completed, all on CPU" << std::endl;
		it+=1;

	}

}

template <typename T>
void Volume_CPU<T>::maxLabelsMGMBlancs(){

	//std::cout << "\tSegmentation MGM for group blancs, all on CPU" << std::endl;
	// update only labels of voxels in group "blancs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_blancs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double* proba_class=(double*) malloc(K*sizeof(double));
	double likelihood=0;
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				if(in_blancs){
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					mean_neighbours=0;
					number_neighbours=0;
					//(-1,0,0)
					if(xn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					mean_neighbours=mean_neighbours/number_neighbours;
					//maximum of probability
					contour=(count_neighbours[0]<number_neighbours);
					if(contour){
						mean_voxel=float(m_classes[0]);
					}else{
						mean_voxel=mean_neighbours;
					}
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];
					for(k=1;k<K;k++){
						contour=(count_neighbours[k]<number_neighbours);
						if(contour){
							mean_voxel=m_classes[k];
						}else{
							mean_voxel=double(mean_neighbours);
						}
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
				    labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);


}

template <typename T>
void Volume_CPU<T>::maxLabelsMGMBlancsFixedContours(){

	//std::cout << "\tSegmentation MGM for group blancs, all on CPU" << std::endl;
	// update only labels of voxels in group "blancs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_blancs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	bool* contours_volume=this->getBooleanContours();
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double* proba_class=(double*) malloc(K*sizeof(double));
	double likelihood=0;
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				if(in_blancs){
					// contour
					contour=contours_volume[adresse_voxel];
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					mean_neighbours=0;
					number_neighbours=0;
					//(-1,0,0)
					if(xn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					mean_neighbours=mean_neighbours/number_neighbours;
					//maximum of probability
					if(contour){
						mean_voxel=float(m_classes[0]);
					}else{
						mean_voxel=mean_neighbours;
					}
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];
					for(k=1;k<K;k++){
						if(contour){
							mean_voxel=m_classes[k];
						}else{
							mean_voxel=double(mean_neighbours);
						}
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
				    labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);


}

template <typename T>
void Volume_CPU<T>::maxLabelsMGMNoirs(){

	//std::cout << "\tSegmentation MGM for group noirs, all on CPU" << std::endl;
	// update only labels of voxels in group "blancs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_noirs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double* proba_class=(double*) malloc(K*sizeof(double));
	double likelihood=0;
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_noirs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_noirs=in_noirs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				in_noirs=!in_noirs;
				if(in_noirs){
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					number_neighbours=0;
					mean_neighbours=0;
					//(-1,0,0)
					if(xn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					mean_neighbours=mean_neighbours/number_neighbours;
					//maximum of probability
					contour=(count_neighbours[0]<number_neighbours);
					if(contour){
						mean_voxel=float(m_classes[0]);
					}else{
						mean_voxel=mean_neighbours;
					}
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];
					for(k=1;k<K;k++){
						contour=(count_neighbours[k]<number_neighbours);
						if(contour){
							mean_voxel=m_classes[k];
						}else{
							mean_voxel=double(mean_neighbours);
						}
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
					labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);

}

template <typename T>
void Volume_CPU<T>::maxLabelsMGMNoirsFixedContours(){

	//std::cout << "\tSegmentation MGM for group noirs, all on CPU" << std::endl;
	// update only labels of voxels in group "blancs"

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k,k_neigh,k_max;//label
	bool in_noirs;//in group or not ?
	double proba_max;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	bool* contours_volume=this->getBooleanContours();
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double* proba_class=(double*) malloc(K*sizeof(double));
	double likelihood=0;
	unsigned int* count_neighbours=(unsigned int*) malloc(K*sizeof(unsigned int));

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				in_noirs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
				in_noirs=in_noirs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
				in_noirs=in_noirs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
				in_noirs=!in_noirs;
				if(in_noirs){
					contour=contours_volume[adresse_voxel];
					// count neighbours in each class
					for(k=0;k<K;k++){
						count_neighbours[k]=0;
					}
					number_neighbours=0;
					mean_neighbours=0;
					//(-1,0,0)
					if(xn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(1,0,0)
					if(xn<Nx-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
						k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,-1,0)
					if(yn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,1,0)
					if(yn<Ny-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
						k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,-1)
					if(zn>0){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					//(0,0,1)
					if(zn<Nz-1){
						number_neighbours+=1;
						mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
						k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
						count_neighbours[k_neigh]+=1;
					}
					mean_neighbours=mean_neighbours/number_neighbours;
					//maximum of probability
					if(contour){
						mean_voxel=float(m_classes[0]);
					}else{
						mean_voxel=mean_neighbours;
					}
					likelihood=energy_singleton[0]-0.5*log(v_classes[0])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[0]);
					proba_class[0]=likelihood+gamma_potts*(double(count_neighbours[0]));
					k_max=0;
					proba_max=proba_class[0];
					for(k=1;k<K;k++){
						if(contour){
							mean_voxel=m_classes[k];
						}else{
							mean_voxel=double(mean_neighbours);
						}
						likelihood=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);
						proba_class[k]=likelihood+gamma_potts*(double(count_neighbours[k]));
						if(proba_max<proba_class[k]){
							k_max=k;
							proba_max=proba_class[k];
						}
					}
					//label
					labels[adresse_voxel]=k_max;
				}
			}
		}
	}

	free(proba_class);
	free(count_neighbours);

}


template <typename T>
double Volume_CPU<T>::computePottsEnergyMGM(){

	std::cout << "\tPotts energy MGM, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;//label
	//compare labels
	unsigned int deltaz_moins,deltaz_plus;
	unsigned int deltay_moins,deltay_plus;
	unsigned int deltax_moins,deltax_plus;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	int number_same_class;//number of neighbours in same class
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double energy_neighbours=0;
	double energy_data=0;

	// initialize
	double potts_energy=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				mean_neighbours=0;
				number_neighbours=0;
				//(-1,0,0)
				deltax_moins=0;
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					if(k==labels[(xn-1)+yn*Nx+zn*Nx*Ny]){
						deltax_moins=1;
					}
				}
				//(1,0,0)
				deltax_plus=0;
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					if(k==labels[(xn+1)+yn*Nx+zn*Nx*Ny]){
						deltax_plus=1;
					}
				}
				//(0,-1,0)
				deltay_moins=0;
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					if(k==labels[xn+(yn-1)*Nx+zn*Nx*Ny]){
						deltay_moins=1;
					}
				}
				//(0,1,0)
				deltay_plus=0;
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					if(k==labels[xn+(yn+1)*Nx+zn*Nx*Ny]){
						deltay_plus=1;
					}
				}
				//(0,0,-1)
				deltaz_moins=0;
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					if(k==labels[xn+yn*Nx+(zn-1)*Nx*Ny]){
						deltaz_moins=1;
					}
				}
				//(0,0,1)
				deltaz_plus=0;
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					if(k==labels[xn+yn*Nx+(zn+1)*Nx*Ny]){
						deltaz_plus=1;
					}
				}

				// mean
				number_same_class=deltax_moins+deltax_plus+deltay_moins+deltay_plus+deltaz_moins+deltaz_plus;
				contour=(number_same_class<number_neighbours);
				mean_neighbours=mean_neighbours/number_neighbours;
				if(contour){
					mean_voxel=m_classes[k];
				}else{
					mean_voxel=double(mean_neighbours);
				}

				energy_neighbours=gamma_potts*double(deltax_moins+deltax_plus+deltay_moins+deltay_plus+deltaz_moins+deltaz_plus);
				energy_data=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);

				potts_energy+=(energy_data+energy_neighbours);
			}
		}
	}

	return potts_energy;

}

template <typename T>
double Volume_CPU<T>::computePottsEnergyMGMFixedContours(){

	std::cout << "\tPotts energy MGM, all on CPU" << std::endl;

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;//label
	//compare labels
	unsigned int deltaz_moins,deltaz_plus;
	unsigned int deltay_moins,deltay_plus;
	unsigned int deltax_moins,deltax_plus;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	double* energy_singleton=this->getEnergySingleton();//energies of singleton
	double gamma_potts=this->getGammaPotts();//Potts coefficient
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;
	int number_same_class;//number of neighbours in same class
	bool* contours_volume=this->getBooleanContours();
	bool contour;// contour
	double mean_voxel;//mean for the voxel

	double energy_neighbours=0;
	double energy_data=0;

	// initialize
	double potts_energy=0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contours_volume[adresse_voxel];
				mean_neighbours=0;
				number_neighbours=0;
				number_same_class=0;
				//(-1,0,0)
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					if(k==labels[(xn-1)+yn*Nx+zn*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}
				//(1,0,0)
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					if(k==labels[(xn+1)+yn*Nx+zn*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}
				//(0,-1,0)
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					if(k==labels[xn+(yn-1)*Nx+zn*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}
				//(0,1,0)
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					if(k==labels[xn+(yn+1)*Nx+zn*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}
				//(0,0,-1)
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					if(k==labels[xn+yn*Nx+(zn-1)*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}
				//(0,0,1)
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					if(k==labels[xn+yn*Nx+(zn+1)*Nx*Ny]){
						number_same_class=number_same_class+1;
					}
				}

				// mean
				mean_neighbours=mean_neighbours/number_neighbours;
				if(contour){
					mean_voxel=m_classes[k];
				}else{
					mean_voxel=double(mean_neighbours);
				}

				energy_neighbours=gamma_potts*double(number_same_class);
				energy_data=energy_singleton[k]-0.5*log(v_classes[k])-0.5*(((double(data_volume[adresse_voxel])-mean_voxel)*(double(data_volume[adresse_voxel])-mean_voxel))/v_classes[k]);

				potts_energy+=(energy_data+energy_neighbours);
			}
		}
	}

	return potts_energy;

}

template <typename T>
void Volume_CPU<T>::selectContoursVolume(){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	T* new_data=this->getVolumeData();

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				if(contour){//select contour
					new_data[adresse_voxel]=data_volume[adresse_voxel];
				}else{
					new_data[adresse_voxel]=0.0;
				}
			}
		}
	}


}

template <typename T>
void Volume_CPU<T>::selectNoContoursVolume(){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	T* new_data=this->getVolumeData();

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				if(contour){//select contour
					new_data[adresse_voxel]=0.0;
				}else{
					new_data[adresse_voxel]=data_volume[adresse_voxel];
				}
			}
		}
	}


}

template <typename T>
void Volume_CPU<T>::selectContoursVolume(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=volume->getVolumeData();//data
	int* labels=volume->getLabels();//labels
	T* new_data=this->getVolumeData();

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				if(contour){//select contour
					new_data[adresse_voxel]=data_volume[adresse_voxel];
				}else{
					new_data[adresse_voxel]=0.0;
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::selectNoContoursVolume(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=volume->getVolumeData();//data
	int* labels=volume->getLabels();//labels
	T* new_data=this->getVolumeData();

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				//(-1,0,0)
				if(xn>0){
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				if(contour){//select contour
					new_data[adresse_voxel]=0.0;
				}else{
					new_data[adresse_voxel]=data_volume[adresse_voxel];
				}
			}
		}
	}


}

template <typename T>
void Volume_CPU<T>::centerMeanVolume(Volume_CPU<T>* volume){// center by the mean of the neighbours

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=volume->getVolumeData();//data
	int* labels=volume->getLabels();//labels
	T* new_data=this->getVolumeData();
	double* v_classes=volume->getVariancesClasses();//variances of the classes
	float mean_neighbours;
	int number_neighbours;

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				mean_neighbours=0.0;
				number_neighbours=0;
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				mean_neighbours=mean_neighbours/number_neighbours;
				if(contour){//select contour
					new_data[adresse_voxel]=0.0;
				}else{
					new_data[adresse_voxel]=T((double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::centerMeanVolumeKnownContours(Volume_CPU<T>* volume){// center by the mean of the neighbours

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;
	int k;

	T* data_volume=volume->getVolumeData();//data
	int* labels=volume->getLabels();//labels
	bool* contours_volume=volume->getBooleanContours();// contours
	T* new_data=this->getVolumeData();
	double* v_classes=volume->getVariancesClasses();//variances of the classes
	float mean_neighbours;
	int number_neighbours;

	bool contour;// contour

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contours_volume[adresse_voxel];
				mean_neighbours=0.0;
				number_neighbours=0;
				if(xn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
				}
				//(1,0,0)
				if(xn<Nx-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
				}
				//(0,-1,0)
				if(yn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
				}
				//(0,1,0)
				if(yn<Ny-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
				}
				//(0,0,-1)
				if(zn>0){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
				}
				//(0,0,1)
				if(zn<Nz-1){
					number_neighbours+=1;
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
				}
				mean_neighbours=mean_neighbours/number_neighbours;
				if(contour){//select contour
					new_data[adresse_voxel]=0.0;
				}else{
					new_data[adresse_voxel]=T((double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}
			}
		}
	}

}

template <typename T>
double Volume_CPU<T>::normalizeCenterClassesVolumeMGM(Volume_CPU<T>* volume,Volume_CPU<T>* volume_mgm){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=volume->getVolumeData();//volume
	T* new_data=this->getVolumeData();//data to center and normalize
	T* data_mgm=volume_mgm->getVolumeData();// volume which has been filtered by median filter
	int* labels=volume->getLabels();//labels
	double* m_classes=volume->getMeansClasses();//means of the classes
	double* v_classes=volume->getVariancesClasses();//variances of the classes
	int K=volume->getNumberClasses();//number of classes
	int k, k_neigh;
	float mean_neighbours;//mean of the neighbours
	float mean_mgm;// mean of the neighbours of filtered volume volume_mgm
	int number_neighbours;
	int num_neighbours2;
	bool contour;// contour
	double norm_cont, norm_interior, total_norm;// norms of contours, regions and sum
	norm_cont=0.0;
	norm_interior=0.0;


	// normalize and center
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				mean_neighbours=0.0;
				mean_mgm=0.0;
				number_neighbours=0;
				num_neighbours2=0;
				//(-1,0,0)
				if(xn>0){
					k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn-1>0)+1+int(yn>0)+int(yn<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[(xn-1)+yn*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
					num_neighbours2=1+int(xn+1<Nx-1)+int(yn>0)+int(yn<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[(xn+1)+yn*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn-1>0)+1+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[xn+(yn-1)*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+1+int(yn+1<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[xn+(yn+1)*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn>0)+int(yn<Ny-1)+int(zn-1>0)+1;
					mean_mgm+=(float(data_mgm[xn+yn*Nx+(zn-1)*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn>0)+int(yn<Ny-1)+1+int(zn+1<Nz-1);
					mean_mgm+=(float(data_mgm[xn+yn*Nx+(zn+1)*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					contour=contour||(k!=k_neigh);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;

				if(contour){
					new_data[adresse_voxel]=T(((double(data_volume[adresse_voxel])-m_classes[k])/v_classes[k])+(double(data_mgm[adresse_voxel])-double(mean_mgm)));
					norm_cont+=((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k])/v_classes[k]);
				}else{
					new_data[adresse_voxel]=T(double(data_mgm[adresse_voxel])-double(mean_mgm));
					norm_interior+=((double(data_volume[adresse_voxel])-double(mean_neighbours))*(double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}

			}
		}
	}

	// sum
	total_norm=norm_cont+norm_interior;
	return total_norm;

}

template <typename T>
double Volume_CPU<T>::normalizeCenterClassesVolumeMGMKnownContours(Volume_CPU<T>* volume,Volume_CPU<T>* volume_mgm){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=volume->getVolumeData();//volume
	T* new_data=this->getVolumeData();//data to center and normalize
	T* data_mgm=volume_mgm->getVolumeData();// volume which has been filtered by median filter
	int* labels=volume->getLabels();//labels
	bool* contours_volume=volume->getBooleanContours();// contours
	double* m_classes=volume->getMeansClasses();//means of the classes
	double* v_classes=volume->getVariancesClasses();//variances of the classes
	int K=volume->getNumberClasses();//number of classes
	int k, k_neigh;
	float mean_neighbours;//mean of the neighbours
	float mean_mgm;// mean of the neighbours of filtered volume volume_mgm
	int number_neighbours;
	int num_neighbours2;
	bool contour;// contour
	double norm_cont, norm_interior, total_norm;// norms of contours, regions and sum
	norm_cont=0.0;
	norm_interior=0.0;


	// normalize and center
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contours_volume[adresse_voxel];
				mean_neighbours=0.0;
				mean_mgm=0.0;
				number_neighbours=0;
				num_neighbours2=0;
				//(-1,0,0)
				if(xn>0){
					k_neigh=labels[(xn-1)+yn*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn-1>0)+1+int(yn>0)+int(yn<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[(xn-1)+yn*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					k_neigh=labels[(xn+1)+yn*Nx+zn*Nx*Ny];
					num_neighbours2=1+int(xn+1<Nx-1)+int(yn>0)+int(yn<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[(xn+1)+yn*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					k_neigh=labels[xn+(yn-1)*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn-1>0)+1+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[xn+(yn-1)*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					k_neigh=labels[xn+(yn+1)*Nx+zn*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+1+int(yn+1<Ny-1)+int(zn>0)+int(zn<Nz-1);
					mean_mgm+=(float(data_mgm[xn+(yn+1)*Nx+zn*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					k_neigh=labels[xn+yn*Nx+(zn-1)*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn>0)+int(yn<Ny-1)+int(zn-1>0)+1;
					mean_mgm+=(float(data_mgm[xn+yn*Nx+(zn-1)*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					k_neigh=labels[xn+yn*Nx+(zn+1)*Nx*Ny];
					num_neighbours2=int(xn>0)+int(xn<Nx-1)+int(yn>0)+int(yn<Ny-1)+1+int(zn+1<Nz-1);
					mean_mgm+=(float(data_mgm[xn+yn*Nx+(zn+1)*Nx*Ny])/num_neighbours2);
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;

				if(contour){
					new_data[adresse_voxel]=T(((double(data_volume[adresse_voxel])-m_classes[k])/v_classes[k])+(double(data_mgm[adresse_voxel])-double(mean_mgm)));
					norm_cont+=((double(data_volume[adresse_voxel])-m_classes[k])*(double(data_volume[adresse_voxel])-m_classes[k])/v_classes[k]);
				}else{
					new_data[adresse_voxel]=T(double(data_mgm[adresse_voxel])-double(mean_mgm));
					norm_interior+=((double(data_volume[adresse_voxel])-double(mean_neighbours))*(double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}

			}
		}
	}

	// sum
	total_norm=norm_cont+norm_interior;
	return total_norm;

}


template <typename T>
double Volume_CPU<T>::normalizeClassesDiffMeansVolume(Volume_CPU<T>* volume, Volume_CPU<T>* volume_means){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_diff=this->getVolumeData();
	T* data_volume=volume->getVolumeData();
	T* data_means=volume_means->getVolumeData();

	int* labels=volume->getLabels();//labels
	double* v_classes=volume->getVariancesClasses();//variances of the classes

	int k;
	double total_norm;
	total_norm=0.0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				data_diff[adresse_voxel]=T((double(data_volume[adresse_voxel]-data_means[adresse_voxel]))/v_classes[k]);
				total_norm=total_norm+T((double(data_volume[adresse_voxel]-data_means[adresse_voxel])*double(data_volume[adresse_voxel]-data_means[adresse_voxel]))/v_classes[k]);
			}
		}
	}

	return total_norm;

}

template <typename T>
double Volume_CPU<T>::sumNormalizedVolumeMGM(){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	int k;
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;//first-order neighbourhood
	bool contour;// contour
	double norm_cont, norm_interior, total_norm;// norms of contours, regions and sum
	norm_cont=0.0;
	norm_interior=0.0;


	// normalize and center
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=false;
				mean_neighbours=0.0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn-1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[(xn+1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn-1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					contour=contour||(k!=labels[xn+(yn+1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn-1)*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					contour=contour||(k!=labels[xn+yn*Nx+(zn+1)*Nx*Ny]);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;

				if(contour){
					norm_cont+=(double(data_volume[adresse_voxel])*double(data_volume[adresse_voxel])/v_classes[k]);
				}else{
					norm_interior+=((double(data_volume[adresse_voxel])-double(mean_neighbours))*(double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}

			}
		}
	}

	// sum
	total_norm=norm_cont+norm_interior;
	return total_norm;

}

template <typename T>
double Volume_CPU<T>::sumNormalizedVolumeMGMKnownContours(){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;
	unsigned long long int adresse_voxel;

	T* data_volume=this->getVolumeData();//data
	int* labels=this->getLabels();//labels
	bool* contours_volume=this->getBooleanContours();
	double* m_classes=this->getMeansClasses();//means of the classes
	double* v_classes=this->getVariancesClasses();//variances of the classes
	int K=this->getNumberClasses();//number of classes
	int k;
	float mean_neighbours;//mean of the neighbours
	int number_neighbours;//first-order neighbourhood
	bool contour;// contour
	double norm_cont, norm_interior, total_norm;// norms of contours, regions and sum
	norm_cont=0.0;
	norm_interior=0.0;


	// normalize and center
	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				adresse_voxel=xn+yn*Nx+zn*Nx*Ny;
				k=labels[adresse_voxel];
				contour=contours_volume[adresse_voxel];
				mean_neighbours=0.0;
				number_neighbours=0;
				//(-1,0,0)
				if(xn>0){
					mean_neighbours+=float(data_volume[(xn-1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(1,0,0)
				if(xn<Nx-1){
					mean_neighbours+=float(data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,-1,0)
				if(yn>0){
					mean_neighbours+=float(data_volume[xn+(yn-1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,1,0)
				if(yn<Ny-1){
					mean_neighbours+=float(data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,-1)
				if(zn>0){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn-1)*Nx*Ny]);
					number_neighbours+=1;
				}
				//(0,0,1)
				if(zn<Nz-1){
					mean_neighbours+=float(data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]);
					number_neighbours+=1;
				}
				mean_neighbours=mean_neighbours/number_neighbours;

				if(contour){
					norm_cont+=(double(data_volume[adresse_voxel])*double(data_volume[adresse_voxel])/v_classes[k]);
				}else{
					norm_interior+=((double(data_volume[adresse_voxel])-double(mean_neighbours))*(double(data_volume[adresse_voxel])-double(mean_neighbours))/v_classes[k]);
				}

			}
		}
	}

	// sum
	total_norm=norm_cont+norm_interior;
	return total_norm;

}

template <typename T>
void Volume_CPU<T>::grad_xplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(xn==(Nx-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[(xn+1)+yn*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_xmoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(xn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[(xn-1)+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_yplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(yn==(Ny-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+(yn+1)*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_ymoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(yn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[xn+(yn-1)*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_zplus(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(zn==(Nz-1)){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+(zn+1)*Nx*Ny]-data_volume[xn+yn*Nx+zn*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::grad_zmoins(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* grad_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				if(zn==0){
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny];
				}else{
					grad_data[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]-data_volume[xn+yn*Nx+(zn-1)*Nx*Ny];
				}
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::sign_volume(Volume_CPU<T>* volume){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* sign_data=this->getVolumeData();//gradient
	T* data_volume=volume->getVolumeData();//data
	T value;
	int signe_volume;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				value=data_volume[xn+yn*Nx+zn*Nx*Ny];
				if(value>0){
					signe_volume=1;
				}else if(value<0){
					signe_volume=-1;
				}else{
					signe_volume=0;
				}
				sign_data[xn+yn*Nx+zn*Nx*Ny]=T(signe_volume);
			}
		}
	}

}

template <typename T>
void Volume_CPU<T>::weightVolume(T* weights){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* data_volume=this->getVolumeData();

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				data_volume[xn+yn*Nx+zn*Nx*Ny]=data_volume[xn+yn*Nx+zn*Nx*Ny]/weights[xn+yn*Nx+zn*Nx*Ny];
			}
		}
	}

}

template <typename T>
double Volume_CPU<T>::sumWeightedVolume(T* weights){

	//dimensions
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();
	unsigned long int xn,yn,zn;

	T* data_volume=this->getVolumeData();
	double sum_weighted=0.0;

	for(zn=0;zn<Nz;zn++){
		for(yn=0;yn<Ny;yn++){
			for(xn=0;xn<Nx;xn++){
				sum_weighted+=(double(data_volume[xn+yn*Nx+zn*Nx*Ny])*double(data_volume[xn+yn*Nx+zn*Nx*Ny])/double(weights[xn+yn*Nx+zn*Nx*Ny]));
			}
		}
	}
	return sum_weighted;
}

#ifdef __CUDACC__
template <typename T>
__host__ void Volume_CPU<T>::copyConstantGPU()
{
	unsigned long int Nx=this->getXVolumePixelNb();
	unsigned long int Ny=this->getYVolumePixelNb();
	unsigned long int Nz=this->getZVolumePixelNb();

	int K = this->getNumberClasses();
	double gamma_potts = this->getGammaPotts();//Potts coefficient

	double* m_classes = this->getMeansClasses();//means of the classes
	double* v_classes = this->getVariancesClasses();//variances of the classes
	double* energy_singleton = this->getEnergySingleton();//energies of singleton

	cudaMemcpyToSymbol(xVolumePixelNb_GPU,&Nx , sizeof(unsigned long int));
	cudaMemcpyToSymbol(yVolumePixelNb_GPU,&Ny , sizeof(unsigned long int));
	cudaMemcpyToSymbol(zVolumePixelNb_GPU,&Nz , sizeof(unsigned long int));

	cudaMemcpyToSymbol( Kclasse , &K, sizeof(int));
	cudaMemcpyToSymbol( gammaPotts , &gamma_potts , sizeof(double) );
	cudaMemcpyToSymbol( meanclasses , m_classes , K*sizeof(double) );
	cudaMemcpyToSymbol( varianceclasses , v_classes , K*sizeof(double) );
	cudaMemcpyToSymbol( energySingleton , energy_singleton , K*sizeof(double) );
	std::cout << "Lancement mémoire constante \n" << std::endl;
}
#endif

/* Volume_CPU_half definition */

Volume_CPU_half::Volume_CPU_half() : Volume_CPU<half>(){}

Volume_CPU_half::Volume_CPU_half(float xVolumeSize, float yVolumeSize, float zVolumeSize, unsigned long int xVolumePixelNb, unsigned long int yVolumePixelNb, unsigned long int zVolumePixelNb, CUDAArchitecture* cudaArchitecture) //: Volume<half>(xVolumeSize, yVolumeSize, zVolumeSize, xVolumePixelNb, yVolumePixelNb, zVolumePixelNb)
{
	this->setXVolumeSize(xVolumeSize);
	this->setYVolumeSize(yVolumeSize);
	this->setZVolumeSize(zVolumeSize);
	this->setXVolumePixelNb(xVolumePixelNb);
	this->setYVolumePixelNb(yVolumePixelNb);
	this->setZVolumePixelNb(zVolumePixelNb);
	this->setXVolumePixelSize(xVolumeSize/xVolumePixelNb);
	this->setYVolumePixelSize(yVolumeSize/yVolumePixelNb);
	this->setZVolumePixelSize(this->getXVolumePixelSize());
	this->setXVolumeCenterPixel(xVolumePixelNb/2.0 - 0.5);
	this->setYVolumeCenterPixel(yVolumePixelNb/2.0 - 0.5);
	this->setZVolumeCenterPixel(zVolumePixelNb/2.0 - 0.5);
	this->setVolumeImage(new Image3D_CPU_half(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb));
}

Volume_CPU_half::~Volume_CPU_half()
{
//		delete (Image3D_CPU_half*)this->getVolumeImage();
}


Volume_CPU_half::Volume_CPU_half(const Volume_CPU_half& volumeToCopy)
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

	this->setVolumeImage(new Image3D_CPU_half(*volumeToCopy.getVolumeImage()));
}

Image3D_CPU_half* Volume_CPU_half::getVolumeImage() const
{
	return (Image3D_CPU_half*)Volume_CPU<half>::getVolumeImage();
}

void Volume_CPU_half::setVolume(float value)
{
	this->getVolumeImage()->setImage(value);
}

void Volume_CPU_half::scalarVolume(float value)
{
	this->getVolumeImage()->scalarImage(value);
}

void Volume_CPU_half::addVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage());
}

void Volume_CPU_half::addVolume(Volume_CPU_half* volume2, float lambda)
{
	this->getVolumeImage()->addImage(volume2->getVolumeImage(),lambda);
}

void Volume_CPU_half::positiveAddVolume(Volume_CPU_half* volume2, float lambda)
{
	this->getVolumeImage()->positiveAddImage(volume2->getVolumeImage(),lambda);
}

void Volume_CPU_half::diffVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->diffImage(volume2->getVolumeImage());
}

void Volume_CPU_half::diffVolume(float lambda, Volume_CPU_half* volume2)
{
	this->getVolumeImage()->diffImage(lambda, volume2->getVolumeImage());
}

void Volume_CPU_half::multVolume(Volume_CPU_half* volume2)
{
	this->getVolumeImage()->multImage(volume2->getVolumeImage());
}

double Volume_CPU_half::scalarProductVolume(Volume_CPU_half* volume2)
{
	return this->getVolumeImage()->scalarProductImage(volume2->getVolumeImage());
}

double Volume_CPU_half::getVolumeL1Norm()
{
	return this->getVolumeImage()->getImageL1Norm();
}

double Volume_CPU_half::getVolumeL2Norm()
{
	return this->getVolumeImage()->getImageL2Norm();
}

double Volume_CPU_half::getVolumeLpNorm(double p)
{
	return this->getVolumeImage()->getImageLpNorm(p);
}

double Volume_CPU_half::getVolumeHuberNorm(double threshold)
{
	return this->getVolumeImage()->getImageHuberNorm(threshold);
}

double Volume_CPU_half::getVolumeMean()
{
	return this->getVolumeImage()->getImageMean();
}

double Volume_CPU_half::getVolumeMeanSquare()
{
	return this->getVolumeImage()->getImageMeanSquare();
}

double Volume_CPU_half::getVolumeStd()
{
	return this->getVolumeImage()->getImageStd();
}

void Volume_CPU_half::getVolumeSign(Volume_CPU_half* signedVolume)
{
	this->getVolumeImage()->getImageSign(signedVolume->getVolumeImage());
}

void Volume_CPU_half::getVolumeAbsPow(Volume_CPU_half* absPowVolume, double p)
{
	this->getVolumeImage()->getImageAbsPow(absPowVolume->getVolumeImage(),p);
}

void Volume_CPU_half::saveVolume(string fileName)
{
	this->getVolumeImage()->saveImage(fileName);
}

void Volume_CPU_half::saveMiddleSliceVolume(string fileName)
{
	this->getVolumeImage()->saveMiddleSliceImage(fileName);
}

void Volume_CPU_half::loadVolume(string fileName)
{
	this->getVolumeImage()->loadImage(fileName);
}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
void Volume_CPU_half::normalizeCenterVolumeMGI(Volume_CPU_half* volume){}

double Volume_CPU_half::sumNormalizedCenteredVolumeMGI(){
	return 0.0;
}

// center each pixel and normalize them (MGI) by the means and the variances of another volume
double Volume_CPU_half::normalizeCenterClassesVolumeMGI(Volume_CPU_half* volume){
	return 0.0;
}

// normalize volume (no centering) by the variances of another volume
void Volume_CPU_half::normalizeVolume(Volume_CPU_half* volume){}

double Volume_CPU_half::sumNormalizedVolume(){
	return 0.0;
}

// normalize volume
void Volume_CPU_half::normalizeVolume(){}

// center volume by the means of another volume
void Volume_CPU_half::centerVolumeMGI(Volume_CPU_half* volume){}

// contours
void Volume_CPU_half::indicateContoursVolume(){}

// means for MGM
void Volume_CPU_half::computeMeansMGMVolume(Volume_CPU_half* volume){}

//Gauss-Markov-Potts : joint MAP (MGI)
void Volume_CPU_half::maxMeansClassesMGI(double m0, double v0){}

void Volume_CPU_half::maxVariancesClassesMGI(double alpha0, double beta0){}

void Volume_CPU_half::maxLabelsMGI(unsigned int numit, double tol){}

void Volume_CPU_half::maxLabelsMGIBlancs(){}

void Volume_CPU_half::maxLabelsMGINoirs(){}

double Volume_CPU_half::computePottsEnergyMGI(){
	return 0;
}

//Gauss-Markov-Potts : joint MAP (MGM)
void Volume_CPU_half::maxMeansClassesMGM(double m0, double v0){}

void Volume_CPU_half::maxVariancesClassesMGM(double alpha0, double beta0){}

void Volume_CPU_half::maxMeansClassesMGMKnownContours(double m0, double v0){}

void Volume_CPU_half::maxVariancesClassesMGMKnownContours(double alpha0, double beta0){}

void Volume_CPU_half::maxLabelsMGM(unsigned int numit, double tol){}

void Volume_CPU_half::maxLabelsMGMBlancs(){}

void Volume_CPU_half::maxLabelsMGMNoirs(){}

double Volume_CPU_half::computePottsEnergyMGM(){
	return 0;
}

void Volume_CPU_half::maxLabelsMGMFixedContours(unsigned int numit, double tol){}

void Volume_CPU_half::maxLabelsMGMBlancsFixedContours(){}

void Volume_CPU_half::maxLabelsMGMNoirsFixedContours(){}

double Volume_CPU_half::computePottsEnergyMGMFixedContours(){
	return 0;
}

void Volume_CPU_half::selectContoursVolume(){}

void Volume_CPU_half::selectNoContoursVolume(){}

void Volume_CPU_half::selectContoursVolume(Volume_CPU_half* volume){}

void Volume_CPU_half::selectNoContoursVolume(Volume_CPU_half* volume){}

void Volume_CPU_half::centerMeanVolume(Volume_CPU_half* volume){}

double Volume_CPU_half::normalizeCenterClassesVolumeMGM(Volume_CPU_half* volume,Volume_CPU_half* volume_mgm){
	return 0.0;
}

double Volume_CPU_half::sumNormalizedVolumeMGM(){
	return 0.0;
}

void Volume_CPU_half::centerMeanVolumeKnownContours(Volume_CPU_half* volume){}

double Volume_CPU_half::normalizeCenterClassesVolumeMGMKnownContours(Volume_CPU_half* volume,Volume_CPU_half* volume_mgm){
	return 0.0;
}

double Volume_CPU_half::normalizeClassesDiffMeansVolume(Volume_CPU_half* volume, Volume_CPU_half* volume_means){
	return 0.0;
}

double Volume_CPU_half::sumNormalizedVolumeMGMKnownContours(){
	return 0.0;
}

void Volume_CPU_half::grad_xplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_xmoins(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_yplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_ymoins(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_zplus(Volume_CPU_half* volume){}

void Volume_CPU_half::grad_zmoins(Volume_CPU_half* volume){}

void Volume_CPU_half::sign_volume(Volume_CPU_half* volume){}

#include "Volume_instances.cu"
#include "Volume_instances_CPU.cu"
