/*
 * Iter3D_algo_Gradient.cu
 *
 *      Author: gac
 */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"


//#include "Iter3D.MGPUcuh"

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::Init1_SG(V<T>* volume,S<T>* sino){}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::Init2_SG(V<T>* volume,V<T>* dJ,S<T>* sino){}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::Init3_SG(S<T>* sino){}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doSimpleGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	high_resolution_clock::time_point t0,t1;
	t0 = high_resolution_clock::now();

	V<T>* dJ = this->create_volume();

	// MedMC Commented It
	//dJ->setVolume(0);

#ifdef COMPUTE_EAM
	V<T>* volume_temp = this->create_volume();
	volume_temp->setVolume(0);
#endif

	S<T>* estimatedSinogram;
	estimatedSinogram = this->create_sinogram3D();

	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double alpha_min = -0.0347711;
	double beta=0.0;
	double eam=0.0;

	unsigned int gradientIterationIdx;

	// Gradient descent
	cout << "Start gradient descent" << endl;
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);
	for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
	{
		cout << "\nGradient iteration n° " << gradientIterationIdx << endl;
		//volume->saveVolume(this->getOutputDirectory() + "avant_proj"+ std::to_string(gradientIterationIdx) + ".v");

		if (gradientIterationIdx == 1)
			this->Init1_SG(volume,estimatedSinogram);
		
		this->getProjector()->doProjection(estimatedSinogram,volume);
		cout << "\tProjection done "  << endl;
		//estimatedSinogram->saveSinogram(this->getOutputDirectory() + "avant_diffsino"+ std::to_string(gradientIterationIdx) + ".s");

		//cout << "\t1 = " << J_MC << endl;
		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

		//cout << "\t2 = " << J_MC << endl;
		//J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		//cout <<"\nSize : "<<estimatedSinogram->getDataSinogramSize() << " MSQ "  << J_MC/estimatedSinogram->getDataSinogramSize()<< " L2 "<< estimatedSinogram->getSinogramL2Norm() <<"\n"<<endl;
		//cout << "\tJ_MC = " << J_MC << endl;


		/* BackProjection */
		//dJ->saveVolume(this->getOutputDirectory() + "avant_BP"+ std::to_string(gradientIterationIdx) + ".v");

			if (gradientIterationIdx == 1)
			  Init2_SG(volume,dJ,estimatedSinogram);
				else
			Init3_SG(estimatedSinogram);

		//dJ->setVolume(0);

		//estimatedSinogram->saveSinogram(this->getOutputDirectory() + "avant_BP"+ std::to_string(gradientIterationIdx) + ".s");

		//cout << "\n Start BackProj Iter : "  << gradientIterationIdx << endl;
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
		cout << "\tBackprojection done "  << endl;

		//delete estimatedSinogram;
		//dJ->saveVolume(this->getOutputDirectory() + "apres_BP"+ std::to_string(gradientIterationIdx) + ".v");
		dJ->scalarVolume(-2.0);
		//dJ->saveVolume(this->getOutputDirectory() + "apres_scalarvolume"+ std::to_string(gradientIterationIdx) + ".v");

		//dJ->saveVolume("vol_debug.v");

		/* Regularization */
		if(this->getLambda() != 0)
		{
			if(this->getKind_Regularizer() == GG)
			{
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			}
			else if(this->getKind_Regularizer() == HUBER)
			{
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			}
		}

		//dJ->saveVolume(this->getOutputDirectory() + "apres_reg"+ std::to_string(gradientIterationIdx) + ".v");
		//cout << "\tLambda=" << this->getLambda() << endl;

		/* Step Computation */
		if(gradientIterationIdx < this->getOptimalStepIterationNb())
		{
			cout << "\tStart Descent Step Computation" << endl;

			//dJ->InitVolume_v1(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),dJ, estimatedSinogram);
			//estimatedSinogram->InitSinogram3D(this->getAcquisition(), this->getDetector() , this->getCUDABProjectionArchitecture(),dJ, estimatedSinogram, 0);
			//dJ->InitVolume(this->getAcquisition(), this->getDetector(), this->getCUDAProjectionArchitecture(),dJ, estimatedSinogram);
			//dJ->loadVolume(this->getConfigCTFile()->getRealVolumeName());

			this->getProjector()->doProjection(estimatedSinogram,dJ);
			//estimatedSinogram->saveSinogram("sino_debug.s");

			normdJ = dJ->getVolumeL2Norm();
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;
			//cout << "\n L2 "<< estimatedSinogram->getSinogramL2Norm() << "\n"<<endl;
			cout << "\tnormdJ = " << normdJ << "  normdJProj = "  << normdJProj << "  normdJProjReg = "  << normdJProjReg << endl;

			alpha=-1.0*(normdJ)/(normdJProj);
			alpha_min = fmax(alpha,alpha_min);
			cout << "\tEnd Descent Step Computation (alpha = " << alpha << ")" << endl;
		}
		else
		{
			alpha = alpha_min;
			cout << "\t(alpha = " << alpha << ")" << endl;
		}
		//			delete estimatedSinogram;
		cout << "\tStart Volume Updating" << endl;
		//volume->saveVolume(this->getOutputDirectory() + "Volume_avant_update"+ std::to_string(gradientIterationIdx) + ".v");
		this->getProjector()->updateVolume(volume,dJ,alpha,this->getPositivity());
		//volume->saveVolume(this->getOutputDirectory() + "apres_update"+ std::to_string(gradientIterationIdx) + ".v");
		cout << "\tEnd Volume Updating" << endl;

		if (gradientIterationIdx == this->getGradientIterationNb())
		{
			this->getBackprojector()->DisableP2P();
		}

#ifdef COMPUTE_MIDDLE_SLICE
		volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
		//volume->saveVolume(this->getOutputDirectory() + "Volume"+ std::to_string(gradientIterationIdx) + ".v");

		//V<T> volume_temp(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
		//volume_temp.setVolume(0);
#ifdef COMPUTE_EAM
		volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
		volume_temp->diffVolume(realVolume);
		eam=volume_temp->getVolumeL1Norm();
#endif
#ifdef COMPUTE_J
		J = J_MC + this->getLambda()*JReg;
		cout << "\tJ = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(gradientIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif
		//			gettimeofday(&end, NULL);
		//			seconds  = end.tv_sec  - startc.tv_sec;
		//			useconds = end.tv_usec - startc.tv_usec;
		//			msecTotal = useconds;
		//			msecTotal/=1000;
		//			msecTotal+=seconds*1000;
		//			cout << "Time iter = " << msecTotal <<  " msec" << endl;
	}


	t1 = high_resolution_clock::now();


	/*error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);*/
	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout <<"Gradient computation :"  << total_ms.count() << "ms\n";
	ofstream myfile;
	myfile.open ("timings.log", ios::app);
	myfile << "ngpu " << this->getConfigComputeArchitectureFile()->getGpuNb_proj() << " nstreams " << this->getConfigComputeArchitectureFile()->getprojectionStreamsNb() << " storage " << this->getConfigComputeArchitectureFile()->getStorage()<< " precision " << this->getConfigComputeArchitectureFile()->getPrecision() << " temps gradient " << total_ms.count() << "ms\n";
	myfile.close();
	// commented

	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	volume->saveVolumeIter(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;


}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doWeightedGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	V<T>* dJ = this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
#ifdef COMPUTE_EAM
	V<T>* volume_temp = this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	volume_temp->setVolume(0);
#endif
	S<T>* estimatedSinogram;
	estimatedSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);

	S<T>* weightedSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);
	stringstream weightedSinogramFile;
	weightedSinogramFile << this->sinogramDirectory << "/weightedSinogram.s";
	weightedSinogram->loadSinogram(weightedSinogramFile.str() );
	cout << "Weight sinogram initialized" << endl;

	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double alpha_min = -999999999999;
	double beta=0.0;
	double eam=0.0;

	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;


	// Gradient descent
	cout << "Start gradient descent " << endl;
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);
	for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
	{
		//			gettimeofday(&startc, NULL);
		cout << "Gradient iteration n° " << gradientIterationIdx << endl;

		/* Projection */
		this->getProjector()->doProjection(estimatedSinogram,volume);
		cout << "Projection done "  << endl;

		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

		J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		cout << "J_MC = " << J_MC << endl;

		estimatedSinogram->multSinogram(weightedSinogram);

		dJ->setVolume(0);
		/* BackProjection */
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
		cout << "Backprojection done "  << endl;

		//delete estimatedSinogram;
		dJ->scalarVolume(-2.0);

		/* Regularization */
		if(this->getLambda() != 0){
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),gradientIterationIdx,this->getOptimalStepIterationNb());
		}

		/* Step Computation */
		if(gradientIterationIdx < this->getOptimalStepIterationNb())
		{
			cout << "Start Descent Step Computation" << endl;
			this->getProjector()->doProjection(estimatedSinogram,dJ);

			normdJ = dJ->getVolumeL2Norm();
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

			cout << "normdJ = " << normdJ << "normdJProj = "  << normdJProj << "normdJProjReg = "  << normdJProjReg << endl;

			alpha=-1.0*(normdJ)/(normdJProj);
			alpha_min = fmax(alpha,alpha_min);
			cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;
		}
		else
		{
			alpha = alpha_min;
			cout << "alpha = " << alpha << ")" << endl;
		}
		//			delete estimatedSinogram;

		cout << "Start Volume Updating" << endl;

		this->getProjector()->updateVolume(volume,dJ,alpha,this->getPositivity());

		cout << "End Volume Updating" << endl;
#ifdef COMPUTE_MIDDLE_SLICE
		volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
#ifdef COMPUTE_EAM
		volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
		volume_temp->diffVolume(realVolume);
		eam=volume_temp->getVolumeL1Norm();
#endif
#ifdef COMPUTE_J
		J = J_MC + lambda*JReg;
		cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(gradientIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif
		//			gettimeofday(&end, NULL);
		//			seconds  = end.tv_sec  - startc.tv_sec;
		//			useconds = end.tv_usec - startc.tv_usec;
		//			msecTotal = useconds;
		//			msecTotal/=1000;
		//			msecTotal+=seconds*1000;
		//			cout << "Time iter = " << msecTotal <<  " msec" << endl;
	}

	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);


	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	//volume->saveVolume(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	delete weightedSinogram;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;

}

template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doConjugateGradient(V<T>* volume,S<T>* realSinogram,V<T>* realVolume)
{
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	V<T>* dJ = this->create_volume();
	dJ->setVolume(0);
#ifdef COMPUTE_EAM
	V<T>* volume_temp = this->create_volume();
	volume_temp->setVolume(0);
#endif
	S<T>* estimatedSinogram;
	estimatedSinogram = this->create_sinogram3D();


	V<T>* olddJ = this->create_volume();
	olddJ->setVolume(0);
	V<T>* p = this->create_volume();
	p->setVolume(0);

	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normolddJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double beta=0.0;
	double eam=0.0;

	unsigned int totalIterationIdx = 0;
	unsigned int globalIterationIdx;
	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;

	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);
	for(globalIterationIdx = 1; globalIterationIdx <= this->getGlobalIterationNb(); globalIterationIdx++)
	{
		cout << "Global iteration n° " << globalIterationIdx << endl;

		// Conjugate Gradient descent
		cout << "Start conjugate gradient descent" << endl;

		// Init Conjugate Gradient descent
		cout << "Initialization conjugate gradient descent" << endl;

		gradientIterationIdx = 1;
		cout << "Gradient iteration n° " << gradientIterationIdx << endl;
		totalIterationIdx = (globalIterationIdx-1)*this->getGradientIterationNb()+gradientIterationIdx;
		cout << "Total iteration n° " << totalIterationIdx << endl;

		this->getProjector()->doProjection(estimatedSinogram, volume);
		estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);
		J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
		cout << "J_MC = " << J_MC << endl;
		/* BackProjection */
		dJ->setVolume(0);
		this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
		dJ->scalarVolume(-2.0);

		/* Regularization */
		if(this->getLambda() != 0){
			if(this->getKind_Regularizer() == GG)
				this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),totalIterationIdx,this->getOptimalStepIterationNb());
			else if(this->getKind_Regularizer() == HUBER)
				this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ, &JReg, &normdJProjReg,this->getLambda(),totalIterationIdx,this->getOptimalStepIterationNb());
		}

		*olddJ=*p;

		/* Step Computation */

		cout << "Start Descent Step Computation" << endl;
		this->getProjector()->doProjection(estimatedSinogram,p);
		normdJ = p->getVolumeL2Norm();
		normolddJ = normdJ;
		normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

		alpha=-1.0*(normdJ)/(normdJProj);
		cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;

		cout << "Start Volume Updating" << endl;
		this->getProjector()->updateVolume(volume,p,alpha,1);
		cout << "End Volume Updating" << endl;
		this->getProjector()->getVolume()->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
		J = J_MC + this->getLambda()*JReg;
		cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
		this->getConfigIterationFile()->updateLogFile(totalIterationIdx,J,J_MC,JReg,alpha,beta,eam);
		cout << "End Initialization conjugate gradient descent" << endl;

		for (gradientIterationIdx = 2 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
		{

			cout << "Gradient iteration n° " << gradientIterationIdx << endl;
			totalIterationIdx = (globalIterationIdx-1)*this->getGradientIterationNb()+gradientIterationIdx;
			cout << "Total iteration n° " << totalIterationIdx << endl;
			/* Projection */
			this->getProjector()->doProjection(estimatedSinogram, volume);
			estimatedSinogram->diffSinogram(realSinogram,estimatedSinogram);

			J_MC = estimatedSinogram->getSinogramMeanSquare()*estimatedSinogram->getDataSinogramSize();
			cout << "J_MC = " << J_MC << endl;

			/* BackProjection */
			dJ->setVolume(0);
			this->getBackprojector()->doBackProjection(dJ,estimatedSinogram);
			dJ->scalarVolume(-2.0);


			/* Regularization */
			if(this->getLambda() != 0){
				if(this->getKind_Regularizer() == GG)
					this->getRegularizer_GG()->getLaplacianRegularizationCriterion(volume, dJ, p, olddJ,&JReg, &normdJProjReg, &normdJ, &normolddJ, &beta,this->getLambda());
				else if(this->getKind_Regularizer() == HUBER)
					this->getRegularizer_Huber()->getLaplacianRegularizationCriterion(volume, dJ,p, olddJ, &JReg, &normdJProjReg,&normdJ, &normolddJ, &beta,this->getLambda());
			}


			cout << "Estimated sinogram initialized" << endl;

			/* Step Computation */

			cout << "Start Descent Step Computation" << endl;

			this->getProjector()->doProjection(estimatedSinogram,p);
			normdJ = p->scalarProductVolume(dJ);
			normdJProj = 2.0*estimatedSinogram->getSinogramL2Norm()+this->getLambda()*normdJProjReg;

			alpha=-1.0*(normdJ)/(normdJProj);
			cout << "End Descent Step Computation (alpha = " << alpha << ")" << endl;

			cout << "Start Volume Updating" << endl;
			this->getProjector()->updateVolume(volume,p,alpha,this->getPositivity());
			cout << "End Volume Updating" << endl;
#ifdef COMPUTE_MIDDLE_SLICE
			volume->saveMiddleSliceVolume(this->getOutputDirectory() + "middleSlice.v");
#endif
#ifdef COMPUTE_EAM
			volume_temp->getVolumeImage()->copyImage3D(volume->getVolumeImage());
			volume_temp->diffVolume(realVolume);
			eam=volume_temp->getVolumeL1Norm();
#endif
#ifdef COMPUTE_J
			J = J_MC + lambda*JReg;
			cout << "J = J_MC + lambda*JReg = " << J << " JReg = " << JReg << endl;
			this->getConfigIterationFile()->updateLogFile(totalIterationIdx,J,J_MC,JReg,alpha,beta,eam);
#endif



		}


	}

	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);


	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	//volume->saveVolume(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete dJ;
	delete estimatedSinogram;
	//delete olddJ;
	//delete p;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;

}


template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
void Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doProjBack(V<T>* volume)
{
	
	cudaError_t error;

	cudaEvent_t start,stop;
	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	high_resolution_clock::time_point t0,t1;
	t0 = high_resolution_clock::now();


	//V<T>* dJ = new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);

	// MedMC Commented It
	//dJ->setVolume(0);

#ifdef COMPUTE_EAM
	V<T>* volume_temp = this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	volume_temp->setVolume(0);
#endif



	S<T>* estimatedSinogram;
	estimatedSinogram = this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb(),this->cudaArchitectureSino);


	/* Global iteration */
	cout << "********** Start 3D Iterative Reconstruction **********" << endl;

	double J_MC = 0.0;
	double J = 0.0;
	double JReg = 0.0;
	double normdJ = 0.0;
	double normdJProj = 0.0;
	double normdJProjReg = 0.0;
	double alpha = 0.0;
	double alpha_min = -0.0347711;
	double beta=0.0;
	double eam=0.0;

	unsigned int gradientIterationIdx;

	//	float msecTotal = 0.0;
	//	struct timeval startc, end;
	//	long seconds, useconds;




	// Gradient descent
	cout << "Start gradient descent" << endl;
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);
	for (gradientIterationIdx = 1 ;gradientIterationIdx <= this->getGradientIterationNb();gradientIterationIdx++)
	{
		//			gettimeofday(&startc, NULL);
		cout << "\nGradient iteration n° " << gradientIterationIdx << endl;

		/* Projection */
		//volume->saveVolume(this->getOutputDirectory() + "avant_proj"+ std::to_string(gradientIterationIdx) + ".v");

		//added
		if (gradientIterationIdx == 1)
			this->Init1_SG(volume,estimatedSinogram);

		this->getProjector()->doProjection(estimatedSinogram,volume);
		cout << "\tProjection done "  << endl;
	
		if (gradientIterationIdx == 1)
		Init2_SG(volume,volume,estimatedSinogram);
			else
		Init3_SG(estimatedSinogram);

		this->getBackprojector()->doBackProjection(volume,estimatedSinogram);
		cout << "\tBackprojection done "  << endl;


		if (gradientIterationIdx == this->getGradientIterationNb())
		{
			this->getBackprojector()->DisableP2P();
		}

	}
	t1 = high_resolution_clock::now();

	/*error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msec, start, stop);
	printf("Time Gradient : %4.3f s\n",msec/1000.0);*/
	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout <<"ProjBack computation :"  << total_ms.count() << "ms\n";
	ofstream myfile;
	myfile.open ("timings.log", ios::app);
	myfile << "ngpu " << this->getConfigComputeArchitectureFile()->getGpuNb_proj() << " nstreams " << this->getConfigComputeArchitectureFile()->getprojectionStreamsNb() << " storage " << this->getConfigComputeArchitectureFile()->getStorage()<< " precision " << this->getConfigComputeArchitectureFile()->getPrecision() << " temps gradient " << total_ms.count() << "ms\n";
	myfile.close();
	// commented

	//	estimatedSinogram->saveSinogram(outputDirectory.str() + "finalSinogram.s");
	volume->saveVolumeIter(this->getOutputDirectory() + "finalVolume.v");

	cout << "********** Start Free Memory **********" << endl;
	delete estimatedSinogram;
	cout << "********** End Free Memory **********" << endl;
	cout << "********** End 3D Iterative Reconstruction **********" << endl;


}

#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"
//#include "Iter3D_instances_MGPU.cu"
