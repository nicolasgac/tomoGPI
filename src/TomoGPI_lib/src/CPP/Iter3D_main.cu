/*
 * Iter3D_main.cu
 *
 *      Author: gac
 */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
//#include "Iter3D.MGPUcuh"


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
int Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doMainIter( int argc, char** argv)
{
	printf("TomoGPI version %s\n",TOMOGPI_VERSION);
	#ifndef FOV_CYLINDER
	printf("FoR (Field of Reconconstruction) cubique\n");
	#else
	printf("FoR (Field of Reconconstruction) cylindrique\n");
	#endif

	V<T>* volume_real;

	if(this->getConfigCTFile()->getBuildRealSinogramFromTiff())
	{Sinogram3D_CPU<T> *realSinogram_tiff;//(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	realSinogram_tiff = &(this->getConfigTiffFile()->createNormalizedMeanNegativeLogSinogram_5dig<T>());
	cout << "createNormalizedMeanNegativeLogSinogram done" << endl;
	realSinogram_tiff->saveSinogram(this->getSinogramDirectory());
	delete realSinogram_tiff;
	cout << "Real sinogram from tiff created" << endl;
	}

	char *op_string = NULL;
	char load=1,save=1;

	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	high_resolution_clock::time_point t0,t1;


	if (getCmdLineArgumentString(argc, (const char**)argv, "no_load", &op_string)){
		load=0;

	}
	if (getCmdLineArgumentString(argc, (const char**)argv, "no_save", &op_string)){
		save=0;

	}
	if (getCmdLineArgumentString(argc, (const char**)argv, "op", &op_string)){
		cudaError_t error;


		if (strcmp(op_string,"proj") == 0){

			S<T>* sino = this->create_sinogram3D();

			//sino->InitSinogram3D(this->getAcquisition(), this->getDetector() , this->getCUDABProjectionArchitecture(),volume, sino, 0);

			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){
				volume_real= this->getVolume(); 

				if (load)
				{
					//volume_real->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume_real, sino);
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
					//volume_real->loadVolume("/espace/tomo_gpi/data3D_0256/phantom3D_0006_shepp_Daouda/1voxel_3D.v");
				}
			}

			cout << "********** WARM UP**************" << endl;
			//this->getProjector()->doProjection(sino,volume_real);
			cout << "********** END WARM UP**********" << endl;

			t0 = high_resolution_clock::now();
			//changed
			this->getProjector()->EnableP2P();
			this->getProjector()->doProjection(sino,volume_real);
			this->getProjector()->DisableP2P();
			t1 = high_resolution_clock::now();

			if (save)
			{
				sino->saveSinogram("proj.s");
			}

		} else if (strcmp(op_string,"back") == 0){
			V<T>* volume ;
			volume = this->getVolume();

			S<T>* realSinogram = this->create_sinogram3D();
			//volume->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, realSinogram);

			if (load)
			{
				//realSinogram->InitSinogram3D(this->getAcquisition(), this->getDetector(), this->getCUDABProjectionArchitecture(),volume,realSinogram, 0);
				realSinogram->loadSinogram(this->getSinogramDirectory());
				//realSinogram->loadSinogram( "Projections_256/1voxel_3D_proj.s"); 
				//realSinogram->saveSinogram("test1.s");
			}

			cout << "********** WARM UP **************" << endl;
			//this->getBackprojector()->doBackProjection(volume,realSinogram);
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			this->getBackprojector()->EnableP2P();
			this->getBackprojector()->doBackProjection(volume,realSinogram);
			this->getBackprojector()->DisableP2P();
			t1 = high_resolution_clock::now();

			if (save)
			{
				volume->saveVolume("back.v");
			}
		}
		/*} else if (strcmp(op_string,"projback") == 0){
			
			S<T>* sino = new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);

			volume= this->getVolume(); 

				sino->InitSinogram3D(this->getAcquisition(), this->getDetector() , this->getCUDABProjectionArchitecture(),volume, sino, 0);
				volume->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, sino);
					volume->loadVolume(this->getConfigCTFile()->getRealVolumeName());
					//cout<<"\n here  "<<this->getCUDAProjectionArchitecture()->getXBlockNb()<< "  " << this->getCUDAProjectionArchitecture()->getYBlockNb() << " " << this->getCUDAProjectionArchitecture()->getZBlockNb() <<endl;
					sino->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
			
			cout << "********** WARM UP **************" << endl;
			this->getBackprojector()->EnableP2P();
			this->getBackprojector()->doBackProjection(volume,sino);
			this->getBackprojector()->DisableP2P();
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			unsigned int gradientIterationIdx;
		
			
			for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
			{
		
			this->getProjector()->doProjection(sino,volume);
			
			this->getBackprojector()->doBackProjection(volume,sino);
			
			}
			
			t1 = high_resolution_clock::now();
			
			if (save)
			{
				volume->saveVolume("back.v");
			}
		}  */
		/*else if(strcmp(op_string,"proj_v2")==0){
			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){

				volume_real= this->getVolume(); //new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
				if (load)
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
			}
			S<T>* sino = new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			cout << "********** WARM UP **************" << endl;
			this->getProjector()->doProjection(sino,volume_real);
			cout << "********** END WARM UP **********" << endl;

			cout << "********** WARM UP **************" << endl;
			this->getProjector()->doProjection_v2(sino,volume_real);
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			this->getProjector()->doProjection_v2(sino,volume_real);
			t1 = high_resolution_clock::now();
			cout << "********** END WARM UP **********" << endl;
			if (save)
				sino->saveSinogram("proj_v2.s");

		} else if(strcmp(op_string,"proj_v0")==0){
			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){

				volume_real= this->getVolume(); //new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
				if (load)
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
			}
			S<T>* sino = new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			cout << "********** WARM UP **************" << endl;
			
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			this->getProjector()->doProjection_v0(sino,volume_real);
			t1 = high_resolution_clock::now();
			if (save)
				sino->saveSinogram("proj_v0.s");


		} /*else if(strcmp(op_string,"coeffdiagHHT_weights")==0){
			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){

				volume_real= this->getVolume(); //new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
				if (load)
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
			}
			S<T>* sino = new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			cout << "********** WARM UP **************" << endl;
			this->getProjector()->weightedCoeffDiagHVHTSFTR(sino,volume_real);
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			this->getProjector()->weightedCoeffDiagHVHTSFTR(sino,volume_real);
			t1 = high_resolution_clock::now();
			if (save)
				sino->saveSinogram("weightedCoeffDiagHVHT.s");*/

		else if (strcmp(op_string,"reg_GG") == 0){
			V<T>* dJ = this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
			dJ->setVolume(0);
			unsigned int gradientIterationIdx=1;
			double JReg = 0.0;
			double normdJProjReg = 0.0;
			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){

				volume_real= this->getVolume(); //new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
				if (load)
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
			}
			cout << "********** WARM UP **************" << endl;
			//error = cudaEventRecord(start, NULL);
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume_real, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			t1 = high_resolution_clock::now();

			//error = cudaEventRecord(stop, NULL);
			//			error = cudaEventSynchronize(stop);
			if (save)
				dJ->saveVolume("GG.v");


		} else if (strcmp(op_string,"laplacien") == 0){
			T kernel_h[3] = {-1,2,-1};
			T kernel_v[3] = {-1,2,-1};
			T kernel_p[3] = {-1,2,-1};
			if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){

				volume_real= this->getVolume(); //new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
				if (load)
					volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
			}
			V<T>* volume_out = this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
			volume_out->setVolume(0);

			C<T> convolver(kernel_h,kernel_v,kernel_p);

			cout << "********** WARM UP **************" << endl;
			convolver.doSeparableConvolution3D(volume_real,volume_out);
			cout << "********** END WARM UP **********" << endl;

			t0 = high_resolution_clock::now();
			convolver.doSeparableConvolution3D(volume_real,volume_out);
			t1 = high_resolution_clock::now();
			if (save)
				volume_out->saveVolume("laplacien.v");

		}  /*else if (strcmp(op_string,"coeffdiagHTH_weights") == 0){
			V<T>* volume ;
			volume = this->getVolume();
			//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION


			S<T>* realSinogram = new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			if (load)
				realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION

			cout << "********** WARM UP **************" << endl;
			this->getBackprojector()->weightedCoeffDiagHTVHSFTR(volume,realSinogram);
			cout << "********** END WARM UP **********" << endl;


			t0 = high_resolution_clock::now();
			this->getBackprojector()->weightedCoeffDiagHTVHSFTR(volume,realSinogram);
			t1 = high_resolution_clock::now();

			if (save)
				volume->saveVolume("weightedCoeffDiagHTVH.v");*/

	
		 else if (strcmp(op_string,"back_fdk") == 0){
			V<T>* volume ;
			volume = this->getVolume();
			//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION


			S<T>* realSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			if (load)
				realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
			this->getBackprojector()->setFdk(1);
			cout << "********** WARM UP **************" << endl;
			this->getBackprojector()->doBackProjection(volume,realSinogram);
			cout << "********** END WARM UP **********" << endl;


			t0 = high_resolution_clock::now();
			this->getBackprojector()->doBackProjection(volume,realSinogram);
			t1 = high_resolution_clock::now();
			if (save)
				volume->saveVolume("back_fdk.v");

		}  else if (strcmp(op_string,"tiff5") == 0){// when tiff names have 5 digits
			Sinogram3D_CPU<T> *realSinogram_tiff;//(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			realSinogram_tiff = &(this->getConfigTiffFile()->createNormalizedMeanNegativeLogSinogram_5dig<T>());
			cout << "createNormalizedMeanNegativeLogSinogram done" << endl;
			//realSinogram_tiff = &(this->getConfigTiffFile()->createSinogram<T>());
			//cout << "createSinogram done" << endl;
			realSinogram_tiff->saveSinogram(this->getSinogramDirectory());
			delete realSinogram_tiff;
			cout << "Real sinogram from tiff created" << endl;

		} else if (strcmp(op_string,"tiff4") == 0){// when tiff names have 4 digits
			Sinogram3D_CPU<T> *realSinogram_tiff;//(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			realSinogram_tiff = &(this->getConfigTiffFile()->createNormalizedMeanNegativeLogSinogram_4dig<T>());
			cout << "createNormalizedMeanNegativeLogSinogram done" << endl;
			//realSinogram_tiff = &(this->getConfigTiffFile()->createSinogram<T>());
			//cout << "createSinogram done" << endl;
			realSinogram_tiff->saveSinogram(this->getSinogramDirectory());
			delete realSinogram_tiff;
			cout << "Real sinogram from tiff created" << endl;
		}
	}
		
	else
	{

	#ifdef COMPUTE_EAM
		if(this->getConfigCTFile()->getRealVolumeName().compare("none")!=0){
			volume_real= this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);// VOLUME OBJECT CREATION
			volume_real->loadVolume(this->getConfigCTFile()->getRealVolumeName());
		}
		#endif

		V<T>* volume ;
		volume = this->getVolume();

		if(this->getConfigCTFile()->getInitVolumeName().compare("none")!=0){
			if (load)
			{
				//volume->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, realSinogram);
				volume->loadVolume(this->getConfigCTFile()->getInitVolumeName());
			}
		}

		cout << "********** WARM UP **************" << endl;
		//	this->getBackprojector()->EnableP2P();
		//		this->getBackprojector()->doBackProjection(volume,realSinogram);
		//		this->getBackprojector()->DisableP2P();
		cout << "********** END WARM UP **********" << endl;

		t0 = high_resolution_clock::now();
		switch(this->getKind_Gradient()){
		case PROJBACK :
		{
			this->doProjBack(volume);
			cout << "ProjBack" << endl;
		}
		break;
		case SIMPLE :
		{
			S<T>*realSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			if (load)
			{
				//volume->InitVolume_InitIter(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, realSinogram);
				//realSinogram->InitSinogram3D(this->getAcquisition(), this->getDetector() , this->getCUDABProjectionArchitecture(),volume, realSinogram, 0);
				//realSinogram->loadSinogram_InitIter(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
				realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
				//realSinogram->MGPUCopy();
			}
			this->doSimpleGradient(volume,realSinogram,volume_real);
			cout << "Simple Gradient" << endl;
		}
		break;
		case CONJUGATE :
		{
			S<T>*realSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
			if (load)
			{
				//volume->InitVolume_InitIter(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),volume, realSinogram);
				//realSinogram->InitSinogram3D(this->getAcquisition(), this->getDetector() , this->getCUDABProjectionArchitecture(),volume, realSinogram, 0);
				realSinogram->loadSinogram(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
				//realSinogram->loadSinogram_InitIter(this->getSinogramDirectory()); // REAL SINOGRAM OBJECT CREATION AND INITIALIZATION
				//realSinogram->MGPUCopy();
			}

			this->doConjugateGradient(volume,realSinogram,volume_real);
			cout << "Conjugate Gradient" << endl;
		}
		break;
		}

		t1 = high_resolution_clock::now();

		if (save)
			volume->saveVolume(this->getOutputDirectory() + "final_Volume.v");

	}
	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout <<"Time computation :"  << total_ms.count() << "ms\n";
	/*ofstream myfile;
	myfile.open ("timings.log", ios::app);
	myfile << "ngpu " << this->getConfigComputeArchitectureFile()->getGpuNb_proj() << " nstreams " << this->getConfigComputeArchitectureFile()->getprojectionStreamsNb() << " storage " << this->getConfigComputeArchitectureFile()->getStorage()<< " precision " << this->getConfigComputeArchitectureFile()->getPrecision() << " temps total" << total_ms.count() << "ms\n";
	myfile.close();*/
	return 0;

}

#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"
//#include "Iter3D_instances_MGPU.cu"