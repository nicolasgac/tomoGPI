/*
 * Projector_half.cu
 *
 * Author: gac
 */

#include "Projector.cuh"
#include "Projector_CPU.cuh"
#include "Projector_GPU.cuh"
#include "GPUConstant.cuh"

#include "projection_ER_kernel_half.cuh"
#include "projection_ER_kernel_half_UM.cuh"

template<typename V, typename S> Projector_half<V,S>::Projector_half() {}

template<typename V, typename S>
Projector_half<V,S>::Projector_half(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture*  cudaprojectionArchitecture, V* volume) : acquisition(acquisition), detector(detector), volume(volume),  cudaprojectionArchitecture(cudaprojectionArchitecture)
{
	float startAngle = acquisition->getStartAngle();
	float focusDetectorDistance = acquisition->getFocusDetectorDistance();
	float zVolumePixelSize = volume->getZVolumePixelSize();
	float vDetectorPixelSize = detector->getVDetectorPixelSize();


	unsigned short projectionNb = acquisition->getProjectionNb();
	this->alphaIOcylinderC = new float[projectionNb];
	this->betaIOcylinderC = new float[projectionNb];
	this->gammaIOcylinderC = vDetectorPixelSize/(focusDetectorDistance*zVolumePixelSize);

	double* phiValueTab = acquisition->getPhiValue();

	for (int p=0;p<projectionNb;p++)
	{
		alphaIOcylinderC[p] = cos(phiValueTab[p]);
		betaIOcylinderC[p] = sin(phiValueTab[p]);
	}
}

template<typename V, typename S>
Projector_half<V,S>::~Projector_half()
{
	delete alphaIOcylinderC;
	delete betaIOcylinderC;
}

template<typename V, typename S>
Acquisition* Projector_half<V,S>::getAcquisition() const
{
	return this->acquisition;
}

template<typename V, typename S>
Detector* Projector_half<V,S>::getDetector() const
{
	return this->detector;
}


template<typename V, typename S>
CUDAProjectionArchitecture* Projector_half<V,S>::getCUDAProjectionArchitecture() const
{
	return this->cudaprojectionArchitecture;
}


template<typename V, typename S>
V* Projector_half<V,S>::getVolume() const
{
	return this->volume;
}

template<typename V, typename S>
void Projector_half<V,S>::setAcquisition(Acquisition* acquisition)
{
	this->acquisition = acquisition;
}

template<typename V, typename S>
void Projector_half<V,S>::setDetector(Detector* detector)
{
	this->detector = detector;
}


template<typename V, typename S>
void Projector_half<V,S>::setCUDAProjectionArchitecture(CUDAProjectionArchitecture*  cudaprojectionArchitecture)
{
	this->cudaprojectionArchitecture =  cudaprojectionArchitecture;
}

template<typename V, typename S>
void Projector_half<V,S>::setVolume(V* volume)
{
	this->volume = volume;
}

template<typename V, typename S>
void Projector_half<V,S>::updateVolume(V* volume_out,V* volume_in, double lambda, int positive)
{
	if(positive)
	{
		volume_out->positiveAddVolume(volume_in, lambda);
	}
	else
	{
		volume_out->addVolume(volume_in, lambda);
	}
}




template <typename V,typename S>
__host__ void Projector_half<V,S>::copyConstantGPU()
{
	unsigned long int projectionNb = (this->getAcquisition())->getProjectionNb();
	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();
	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();

	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	unsigned long int uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	unsigned long int vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();

	cudaMemcpyToSymbol(xVolumeCenterPixel_GPU,&xVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(yVolumeCenterPixel_GPU,&yVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(zVolumeCenterPixel_GPU,&zVolumeCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(xVolumePixelSize_GPU,&xVolumePixelSize,sizeof(float));
	cudaMemcpyToSymbol(xVolumePixelNb_GPU,&xVolumePixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(yVolumePixelNb_GPU,&yVolumePixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(zVolumePixelNb_GPU,&zVolumePixelNb,sizeof(unsigned long int));

	cudaMemcpyToSymbol(focusDetectorDistance_GPU,&fdd,sizeof(float));
	cudaMemcpyToSymbol(focusObjectDistance_GPU,&fod,sizeof(float));

	cudaMemcpyToSymbol(uDetectorCenterPixel_GPU,&uDetectorCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(vDetectorCenterPixel_GPU,&vDetectorCenterPixel,sizeof(float));
	cudaMemcpyToSymbol(uDetectorPixelSize_GPU,&uDetectorPixelSize,sizeof(float));
	cudaMemcpyToSymbol(vDetectorPixelSize_GPU,&vDetectorPixelSize,sizeof(float));
	cudaMemcpyToSymbol(uDetectorPixelNb_GPU,&uDetectorPixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(vDetectorPixelNb_GPU,&vDetectorPixelNb,sizeof(unsigned long int));
	cudaMemcpyToSymbol(projectionNb_GPU,&projectionNb,sizeof(unsigned long int));

	cudaMemcpyToSymbol(alphaIOcylinderC_GPU,this->alphaIOcylinderC,projectionNb*sizeof(float));
	cudaMemcpyToSymbol(betaIOcylinderC_GPU,this->betaIOcylinderC,projectionNb*sizeof(float));
	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&this->gammaIOcylinderC,sizeof(float));
}


/* RegularSamplingProjector definition */
RegularSamplingProjector_CPU_half::RegularSamplingProjector_CPU_half(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture*  cudaprojectionArchitecture, Volume_CPU_half* volume) : Projector_half<Volume_CPU_half,Sinogram3D_CPU_half>(acquisition, detector,cudaprojectionArchitecture,volume){}

RegularSamplingProjector_CPU_half::RegularSamplingProjector_CPU_half() : Projector_half<Volume_CPU_half,Sinogram3D_CPU_half>(){}

RegularSamplingProjector_CPU_half::~RegularSamplingProjector_CPU_half(){}

void RegularSamplingProjector_CPU_half::doProjection(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume)
{
	std::cout << "Regular Sampling Projection (half precision)" << std::endl;


	TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half>* plan;

	std::thread **threadID;

	this->setVolume(volume);

	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
	float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();



	unsigned long long int size_sinogram;
	int gpuNb = this->getCUDAProjectionArchitecture()->getComputingUnitNb();
	int nstreams=this->getCUDAProjectionArchitecture()->getProjectionStreamsNb();

	this->getCUDAProjectionArchitecture()->setYThreadNb(1);
	this->getCUDAProjectionArchitecture()->setXBlockNb(estimatedSinogram->getUSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getXThreadNb());
	this->getCUDAProjectionArchitecture()->setYBlockNb(estimatedSinogram->getVSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getYThreadNb()/gpuNb);
	this->getCUDAProjectionArchitecture()->setZBlockNb(estimatedSinogram->getProjectionSinogramNb()/this->getCUDAProjectionArchitecture()->getProjectionThreadNb());

	unsigned long long int uSinogramPixelNb = this->getCUDAProjectionArchitecture()->getXThreadNb()*this->getCUDAProjectionArchitecture()->getXBlockNb();
	unsigned long long int vSinogramPixelNb = this->getCUDAProjectionArchitecture()->getYThreadNb()*this->getCUDAProjectionArchitecture()->getYBlockNb();
	unsigned long long int projectionSinogramNb = this->getCUDAProjectionArchitecture()->getProjectionThreadNb()*this->getCUDAProjectionArchitecture()->getZBlockNb();




	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();
	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;

	struct cudaDeviceProp prop_device;
	cudaGetDeviceProperties(&prop_device,0);//propriétés du device 0
	unsigned int nb_bloc_v_par_device=1;

	float taille_bloc_allocation,taille_allocation_vol,taille_allocation_sino,ratio_bloc_SDRAM;
	float taille_allocation,taille_SDRAM,ratio_allocation_SDRAM;
	taille_SDRAM=(float)prop_device.totalGlobalMem;
	taille_allocation_vol=(size_t)sizeof(half)*(size_t)xVolumePixelNb*yVolumePixelNb*(size_t)zVolumePixelNb;//il faut allouer 1 volume
	taille_allocation_sino=(size_t)sizeof(half)*size_sinogram*(size_t)nstreams;//il faut allouer 1 volume
	taille_allocation=taille_allocation_vol+taille_allocation_sino;
	ratio_allocation_SDRAM=taille_allocation/taille_SDRAM;
	printf("allocation : %.2f Go (vol %.2f Go sino %.2f Go) SDRAM : %.2f Go ratio :%.2f\n",taille_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_vol/gpuNb)/((1024.0*1024.0*1024.0)),taille_allocation_sino/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_allocation_SDRAM);


	printf("nb_blocs_v_par_device %d ",nb_bloc_v_par_device);
	taille_bloc_allocation=taille_allocation_vol/((float)nb_bloc_v_par_device*(float)gpuNb)+taille_allocation_sino;
	ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;

	int nb2 = getCUDAProjectionArchitecture()->getYBlockNb();
	while(taille_allocation_sino/taille_SDRAM>=0.5)
	{
		if (nb2>1)
		{
			nb2/=2;
			getCUDAProjectionArchitecture()->setYBlockNb(nb2);
		}
		else
			nstreams/=2;

		vSinogramPixelNb = this->getCUDAProjectionArchitecture()->getYThreadNb()*this->getCUDAProjectionArchitecture()->getYBlockNb();

		size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
		taille_allocation_sino=(size_t)sizeof(half)*size_sinogram*(size_t)nstreams;

	}

	while(ratio_bloc_SDRAM>=0.7)
	{
		nb_bloc_v_par_device*=2;
		printf("%d \n",nb_bloc_v_par_device);
		taille_bloc_allocation=taille_allocation_vol/((float)nb_bloc_v_par_device*(float)gpuNb)+taille_allocation_sino;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	}

	printf("\n");
	printf("allocation par bloc : %.2f Go (vol %.2f Go sino %.2f Go) SDRAM : %.2f Go ratio :%.2f\n",taille_bloc_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_vol/(nb_bloc_v_par_device*gpuNb))/((1024.0*1024.0*1024.0)),taille_allocation_sino/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_bloc_SDRAM);

	//nb_bloc_z_par_device=2;
	unsigned int N_vn_par_carte;
	unsigned int N_vn_par_solverthread;
	unsigned int *num_bloc;
	num_bloc=(unsigned int*)malloc(sizeof(unsigned int)*nb_bloc_v_par_device*gpuNb);
	unsigned int *num_device;
	num_device=(unsigned int*)malloc(sizeof(unsigned int)*gpuNb);

	N_vn_par_carte=estimatedSinogram->getVSinogramPixelNb()/(gpuNb);
	N_vn_par_solverthread=N_vn_par_carte/(nb_bloc_v_par_device);

	int N_ligne_par_carte;
	int N_vn_par_kernel;
	N_vn_par_kernel=this->getCUDAProjectionArchitecture()->getYBlockNb();
	N_ligne_par_carte=(int)(N_vn_par_solverthread/N_vn_par_kernel);

	while (N_ligne_par_carte%nstreams!=0)
	{
		nstreams/=2;
	}

	this->getCUDAProjectionArchitecture()->setProjectionStreamsNb(nstreams);


	printf("N_vn_par_carte %d N_vn_par_solverthread %d nb_bloc_v_par_device %d \n",N_vn_par_carte,N_vn_par_solverthread,nb_bloc_v_par_device);
	cudaEvent_t *start_thread;
	cudaEvent_t *stop_thread;

	start_thread=(cudaEvent_t*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(cudaEvent_t));
	stop_thread=(cudaEvent_t*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(cudaEvent_t));

	plan=(TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half>*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half>));
	//threadID=(CUTThread *)malloc(gpuNb*nb_bloc_v_par_device*sizeof(CUTThread));
	threadID=(std::thread **)malloc(gpuNb*sizeof(std::thread *));

	for(int device=0;device<gpuNb;device++)
	{
		num_device[device]=device;

		cout << "********** Start Constant Copy **********" << endl;
		cout << "Projection Constant Copy on device n° " << device << endl;
		cudaSetDevice(device);
		this->copyConstantGPU();
		cout << "********** End Projection Constant Copy **********" << endl; for (int n=0;n<nb_bloc_v_par_device;n++){
			if (device%2==1)
				num_bloc[n+device*nb_bloc_v_par_device]=(nb_bloc_v_par_device-1)-n;
			else
				num_bloc[n+device*nb_bloc_v_par_device]=n;

			printf("n %d device %d num_device %d n+device*nb_bloc_v_par_device %d num_bloc %d\n",n,device,num_device[device],n+device*nb_bloc_v_par_device,num_bloc[n+device*nb_bloc_v_par_device]);
			checkCudaErrors(cudaEventCreate(start_thread+n+device*nb_bloc_v_par_device));
			checkCudaErrors(cudaEventCreate(stop_thread+n+device*nb_bloc_v_par_device));

			plan[n+device*nb_bloc_v_par_device].device=device;
			plan[n+device*nb_bloc_v_par_device].volume_h=this->getVolume();
			plan[n+device*nb_bloc_v_par_device].sinogram_h=estimatedSinogram;
			plan[n+device*nb_bloc_v_par_device].acquisition=this->getAcquisition();
			plan[n+device*nb_bloc_v_par_device].detector=this->getDetector();
			plan[n+device*nb_bloc_v_par_device].cudaprojectionArchitecture=this->getCUDAProjectionArchitecture();
			plan[n+device*nb_bloc_v_par_device].phi_start=0;

			plan[n+device*nb_bloc_v_par_device].N_vn_par_carte=N_vn_par_carte;
			plan[n+device*nb_bloc_v_par_device].N_vn_par_solverthread=N_vn_par_solverthread;
		}
	}

	for (int n=0;n<nb_bloc_v_par_device;n++){
		for(int device=0;device<gpuNb;device++){
			int vn_start,vn_prime_start,vn_prime_stop,zn_start,zn_stop;
			float zn_prime_start,zn_prime_stop;

			vn_start=num_bloc[n+device*nb_bloc_v_par_device]*N_vn_par_solverthread+num_device[device]*N_vn_par_carte;
			vn_prime_start=vn_start-vDetectorCenterPixel;
			vn_prime_stop=vn_prime_start+N_vn_par_solverthread; if(vn_prime_start>=0){
				zn_prime_start=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			}
			else
			{
				zn_prime_start=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			}


			if(vn_prime_stop>=0){
				zn_prime_stop=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			}
			else
			{
				zn_prime_stop=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			}

			zn_stop=((int)zn_prime_stop+1)+zVolumeCenterPixel;

			if (zn_stop>=zVolumePixelNb)
				zn_stop=zVolumePixelNb-1;

			zn_start=((int)zn_prime_start-1)+zVolumeCenterPixel;

			if (zn_start<0)
				zn_start=0;

			//printf("device %d bloc %d vn_start %d vn_prime_start %d vn_prime_stop %d zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_solverthread %d\n",device,n,vn_start,vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1);

			plan[n+device*nb_bloc_v_par_device].vn_start=vn_start; plan[n+device*nb_bloc_v_par_device].zn_start=zn_start; plan[n+device*nb_bloc_v_par_device].N_zn_par_solverthread=(zn_stop-zn_start)+1;

			//threadID[device+n*nb_bloc_v_par_device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_v_par_device));
			threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_v_par_device));
		}
		//cutWaitForThreads(threadID+n*nb_bloc_v_par_device,gpuNb);
		for (int i = 0; i < gpuNb; i++)
		{
			if (threadID[i]->joinable()){
				//std::cout << i <<" joined:\n" << endl;
				threadID[i]->join();
			}

		}
	}

	for (int i = 0; i < gpuNb; i++)
	{
		delete threadID[i];
	}

	free(threadID);
	free(plan);
	free(start_thread);
	free(stop_thread);
	free(num_bloc);
	free(num_device);
}


void RegularSamplingProjector_CPU_half::doProjectionSFTR(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume)
{

}

void RegularSamplingProjector_CPU_half::doProjectionSFTR_2kernels(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume)
{

}


void RegularSamplingProjector_CPU_half::doProjectionSFTR_opti(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume)
{

}

void RegularSamplingProjector_CPU_half::weightedCoeffDiagHVHTSFTR(Sinogram3D_CPU_half* coeffDiag,Volume_CPU_half* weights){

}


void RegularSamplingProjector_CPU_half::weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_CPU_half* coeffDiag,Volume_CPU_half* weights){

}

void RegularSamplingProjector_CPU_half::doProjectionSFTR_allCPU(Sinogram3D_CPU_half* estimatedSinogram,Volume_CPU_half *volume)
{

}


CUT_THREADPROC RegularSamplingProjector_CPU_half::solverThread(TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half> *plan)
{
	unsigned long long int size_sinogram;
	int vn_prime_start,vn_prime_stop,vn_start,vn_start_old,zn_start,zn_stop,zn_stop_old,N_ligne_par_carte,N_vn_par_kernel,N_vn_restant,kligne;
	float zn_prime_start,zn_prime_stop;

	float fdd = plan->acquisition->getFocusDetectorDistance();
	float fod = plan->acquisition->getFocusObjectDistance();
	float vDetectorPixelSize = plan->detector->getVDetectorPixelSize(); float vDetectorCenterPixel = plan->detector->getVDetectorCenterPixel();

	float xVolumePixelSize = plan->volume_h->getXVolumePixelSize();
	float zVolumeCenterPixel = plan->volume_h->getZVolumeCenterPixel();
	unsigned long int xVolumePixelNb = plan->volume_h->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = plan->volume_h->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = plan->volume_h->getZVolumePixelNb();

	half* volumeData = plan->volume_h->getVolumeData();
	half* estimatedSinogramData = plan->sinogram_h->getDataSinogram(); half** sinogram_df;
	cudaStream_t *streams;
	int nstreams=plan->cudaprojectionArchitecture->getProjectionStreamsNb(); //Set device
	checkCudaErrors(cudaSetDevice(plan->device));
	printf("GPU : %d \n",plan->device);

	cudaEvent_t start_solverthread,stop_solverthread;
	checkCudaErrors(cudaEventCreate(&start_solverthread));
	checkCudaErrors(cudaEventCreate(&stop_solverthread));
	checkCudaErrors(cudaEventRecord(start_solverthread, NULL));

	N_vn_par_kernel=plan->cudaprojectionArchitecture->getYBlockNb();
	N_ligne_par_carte=(int)(plan->N_vn_par_solverthread/N_vn_par_kernel);


	streams = (cudaStream_t*) malloc((nstreams)*sizeof(cudaStream_t));
	for(int i=0; i<nstreams ; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])) ;

	cudaEvent_t *event;
	event=(cudaEvent_t *)malloc(nstreams*sizeof(cudaEvent_t));
	for(int i=0;i<nstreams;i++)
		checkCudaErrors(cudaEventCreate(event+i));

	N_vn_restant=((float)(plan->N_vn_par_solverthread)/N_vn_par_kernel)-N_ligne_par_carte;
	if(N_vn_restant>0)
	{
		N_ligne_par_carte+=1;
	}

	dim3 dimBlock(plan->cudaprojectionArchitecture->getXThreadNb(),plan->cudaprojectionArchitecture->getYThreadNb(),plan->cudaprojectionArchitecture->getProjectionThreadNb());
	dim3 dimGrid(plan->cudaprojectionArchitecture->getXBlockNb(), plan->cudaprojectionArchitecture->getYBlockNb(), plan->cudaprojectionArchitecture->getZBlockNb());

	//Decoupage en thread
	unsigned long long int uSinogramPixelNb = plan->cudaprojectionArchitecture->getXThreadNb()*plan->cudaprojectionArchitecture->getXBlockNb();
	unsigned long long int vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
	unsigned long long int projectionSinogramNb = plan->cudaprojectionArchitecture->getProjectionThreadNb()*plan->cudaprojectionArchitecture->getZBlockNb();

	//allocate device memory for sinogram result
	size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;

	sinogram_df=(half**)malloc(sizeof(half*)*nstreams);
	for(int i=0;i<nstreams;i++)
		checkCudaErrors(cudaMalloc((void**) &(sinogram_df[i]), (unsigned long long int)sizeof(half)*(unsigned long long int)size_sinogram));

	//VOLUME MEMORY ALLOCATION
	//Create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(half), 0, 0, 0, cudaChannelFormatKindFloat);
	struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	volume_cu_array_size= make_cudaExtent((size_t)plan->volume_h->getXVolumePixelNb(),(size_t)plan->volume_h->getYVolumePixelNb(),(size_t)plan->N_zn_par_solverthread);
	checkCudaErrors( cudaMalloc3DArray(&volume_cu_array, &channelDesc,volume_cu_array_size ) );

	cudaMemcpy3DParms *copyParams ;
	copyParams=(cudaMemcpy3DParms *)malloc(nstreams*sizeof(cudaMemcpy3DParms));
	for(int i=0;i<nstreams;i++){
		//copyParams[i]=(const struct cudaMemcpy3DParms){0};
		copyParams[i].dstArray = volume_cu_array;
		copyParams[i].kind = cudaMemcpyHostToDevice;
	}

	// set texture parameters
	volume_tex.addressMode[0] = cudaAddressModeBorder;
	volume_tex.addressMode[1] = cudaAddressModeBorder;
	volume_tex.addressMode[2] = cudaAddressModeBorder;
	volume_tex.filterMode = cudaFilterModeLinear;
	volume_tex.normalized = false; // access with normalized texture coordinates
	// Bind the array to the 3D texture
	checkCudaErrors(cudaBindTextureToArray(volume_tex,volume_cu_array,channelDesc));

	for(kligne=0;kligne<((N_ligne_par_carte/nstreams));kligne++)
	{
		int i=0;

		zn_stop_old=zn_stop;
		vn_start=plan->vn_start+(i+kligne*nstreams)*N_vn_par_kernel;
		vn_prime_start=vn_start-vDetectorCenterPixel;
		vn_prime_stop=vn_prime_start+N_vn_par_kernel;

		if(vn_prime_stop>=0)
			zn_prime_stop=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
		else
			zn_prime_stop=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);

		zn_stop=((int)zn_prime_stop+1)+zVolumeCenterPixel;

		if (zn_stop>zVolumePixelNb)
			zn_stop=zVolumePixelNb-1;

		if (kligne!=0)
			zn_start=zn_stop_old+1;
		else {
			if(vn_prime_start>=0)
				zn_prime_start=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			else
				zn_prime_start=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);

			zn_start=((int)zn_prime_start-1)+zVolumeCenterPixel;

			if (zn_start<0)
				zn_start=0;
			if ((zn_start-plan->zn_start)<0)
				zn_start=plan->zn_start;
		}
		printf("device %d vn_start %d vn_prime_start %d vn_prime_stop %d zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_kernel %d\n",plan->device,vn_start,vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1);



		copyParams[i].srcPos = make_cudaPos(0,0,0);//make_cudaPos(0,plan->vn_start,phi);
		copyParams[i].dstPos = make_cudaPos(0,0,(size_t)(zn_start-plan->zn_start));

		printf("zn_start-plan->zn_start %d (zn_stop-zn_start)+1 %d\n",zn_start-plan->zn_start,(zn_stop-zn_start)+1);
		size_t coord_d;
		coord_d=(size_t)zn_start*(size_t)xVolumePixelNb*(size_t)yVolumePixelNb;

		copyParams[i].srcPtr = make_cudaPitchedPtr(&volumeData[coord_d],(size_t)xVolumePixelNb* (size_t)sizeof(half),(size_t)xVolumePixelNb,(size_t)yVolumePixelNb);

		copyParams[i].extent = make_cudaExtent((size_t)xVolumePixelNb,(size_t)yVolumePixelNb,(size_t)((zn_stop-zn_start)+1));

		if (((zn_stop-zn_start)+1)>0)
			checkCudaErrors(cudaMemcpy3DAsync(&copyParams[i]));

		unsigned long int uSinoPixelNb = plan->sinogram_h->getUSinogramPixelNb();
		unsigned long int vSinoPixelNb = plan->sinogram_h->getVSinogramPixelNb();
		unsigned long int phiSinoNb = plan->sinogram_h->getProjectionSinogramNb();

		for(i=0 ; i < nstreams-1 ; i++)
		{
			checkCudaErrors(cudaEventRecord (event[i], streams[i]));
			projection_ERB_kernel_v1_half<<< dimGrid, dimBlock,0,streams[i]>>>((unsigned short*)(sinogram_df[i]),vn_start,plan->zn_start);

			if (i<nstreams)
			{
				zn_stop_old=zn_stop;
				vn_start_old=vn_start;
				vn_start=plan->vn_start+(i+1+kligne*nstreams)*N_vn_par_kernel;
				vn_prime_start=vn_start-vDetectorCenterPixel;
				vn_prime_stop=vn_prime_start+N_vn_par_kernel;


				if(vn_prime_stop>=0)
					zn_prime_stop=(fod+xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/(fdd*xVolumePixelSize);
				else
					zn_prime_stop=(fod-xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/(fdd*xVolumePixelSize);

				zn_stop=((int)zn_prime_stop+1)+zVolumeCenterPixel;

				if (zn_stop>=zVolumePixelNb)
					zn_stop=zVolumePixelNb-1;

				zn_start=zn_stop_old+1;

				printf("device %d vn_start %d vn_start_old %d vn_prime_start %d vn_prime_stop %d zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_kernel %d\n",plan->device,vn_start,vn_start_old,vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1); copyParams[i+1].srcPos = make_cudaPos(0,0,0);//make_cudaPos(0,plan->vn_start,phi);
				copyParams[i+1].dstPos = make_cudaPos(0,0,zn_start-plan->zn_start);

				printf("zn_start-plan->zn_start %d (zn_stop-zn_start)+1 %d\n",zn_start-plan->zn_start,(zn_stop-zn_start)+1);
				size_t coord_d;
				coord_d=(size_t)zn_start*(size_t)xVolumePixelNb*(size_t)yVolumePixelNb;

				copyParams[i+1].srcPtr = make_cudaPitchedPtr(&(volumeData[coord_d]), xVolumePixelNb*sizeof(half), xVolumePixelNb, yVolumePixelNb);

				copyParams[i+1].extent = make_cudaExtent((unsigned long long int)xVolumePixelNb,(unsigned long long int)yVolumePixelNb,(unsigned long long int)(zn_stop-zn_start)+1);


				if (((zn_stop-zn_start)+1)>0){

					checkCudaErrors(cudaStreamWaitEvent( streams[i+1], event[i],0 )); checkCudaErrors(cudaMemcpy3DAsync(copyParams+i+1,streams[i+1]));
				}
				checkCudaErrors(cudaEventRecord (event[i+1], streams[i+1]));
			}



			for (int phi=0;phi<phiSinoNb;phi++){
				unsigned long long int coord_sino_h;
				unsigned long long int coord_sino_d;
				coord_sino_h=(unsigned long int)(vn_start_old*uSinoPixelNb)+(unsigned long int)(phi)*(unsigned long int)(uSinoPixelNb)*(unsigned long int)(vSinoPixelNb);
				coord_sino_d=(unsigned long int)(phi)*(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb;
				checkCudaErrors(cudaMemcpyAsync(estimatedSinogramData+coord_sino_h,sinogram_df[i]+coord_sino_d,(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb*sizeof(half),cudaMemcpyDeviceToHost,streams[i]));
			}
		}

		if (i<nstreams){
			projection_ERB_kernel_v1_half<<< dimGrid, dimBlock,0,streams[i]>>>((unsigned short*)(sinogram_df[i]),vn_start,plan->zn_start);

			for (int phi=0;phi<phiSinoNb;phi++){
				unsigned long long int coord_sino_h;
				unsigned long long int coord_sino_d;
				coord_sino_h=(unsigned long int)(vn_start*uSinoPixelNb)+(unsigned long int)(phi)*(unsigned long int)(uSinoPixelNb)*(unsigned long int)(vSinoPixelNb);
				coord_sino_d=(unsigned long int)(phi)*(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb;
				checkCudaErrors(cudaMemcpyAsync(estimatedSinogramData+coord_sino_h,sinogram_df[i]+coord_sino_d,(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb*sizeof(half),cudaMemcpyDeviceToHost,streams[i]));
			}
		}
	}

	for(int i=0 ; i < nstreams ; i++)
	{
		cudaStreamSynchronize(streams[i]);
		cudaEventDestroy(event[i]);
	}

	for(int i=0;i<nstreams;i++)
		checkCudaErrors(cudaFree(sinogram_df[i]));

	free(sinogram_df);
	checkCudaErrors(cudaFreeArray(volume_cu_array));


	for(int i = 0 ; i < nstreams ; i++)
	{
		checkCudaErrors(cudaStreamDestroy(streams[i]));

	}

	free(streams);

	free(copyParams);

	checkCudaErrors(cudaEventRecord(stop_solverthread, NULL));
	checkCudaErrors(cudaEventSynchronize(stop_solverthread));

	cudaEventDestroy(start_solverthread);
	cudaEventDestroy(stop_solverthread);

	CUT_THREADEND;
}

//CUT_THREADPROC RegularSamplingProjector_CPU_half::solverThread(TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half> *plan)
//{
// unsigned long int size_sinogram;
// unsigned long int size_tuile;
//
// unsigned int kv_start,kv_stop,kv_max;
// unsigned int semi_plan_z;
// float nb_kv_par_device;
// Sinogram3D_CPU_half* sinogram_temp;
// int gpuNb = plan->cudaprojectionArchitecture->getComputingUnitNb();
//
// half* sinogram_d;
// cudaStream_t *streams;
// int nstreams=4;
//
// //Set device
// checkCudaErrors(cudaSetDevice(plan->device));
// printf("GPU : %d \n",plan->device);
//
// int phi_par_kernel = plan->sinogram_h->getProjectionSinogramNb()/plan->cudaprojectionArchitecture->getProjectionThreadNb();
//
// while (phi_par_kernel%nstreams!=0){
// nstreams/=2;
// }
//
// printf("Streams Number : %d\n",nstreams);
//
//
// streams = (cudaStream_t*) malloc((nstreams)*sizeof(cudaStream_t));
// for(int i=0; i<nstreams ; i++)
// checkCudaErrors(cudaStreamCreate(&streams[i])) ;
//
//
// //Decoupage en thread
// //Parallélisation multi-GPU sur kv
// unsigned long int uSinogramPixelNb = plan->cudaprojectionArchitecture->getXThreadNb()*plan->cudaprojectionArchitecture->getXBlockNb();
// unsigned long int vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
// unsigned long int projectionSinogramNb = plan->sinogram_h->getProjectionSinogramNb();
//
// kv_max=plan->sinogram_h->getVSinogramPixelNb()/vSinogramPixelNb;
// nb_kv_par_device=(float)kv_max/(float)gpuNb;
//
// if (gpuNb==1){
// if (nb_kv_par_device==1)
// {
// plan->cudaprojectionArchitecture->setYBlockNb(plan->cudaprojectionArchitecture->getYBlockNb()/2);
// vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
// }
// }
// else {
// if (nb_kv_par_device<1)
// {
// plan->cudaprojectionArchitecture->setYBlockNb(plan->cudaprojectionArchitecture->getYBlockNb()*nb_kv_par_device);
// vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
// }
// }
// nb_kv_par_device=(float)kv_max/(float)gpuNb;
// kv_max=plan->sinogram_h->getVSinogramPixelNb()/vSinogramPixelNb;
//
// //Allocation du sinogramme temp lié à un kernel (sauf pour phi)
// if (plan->sinogram_h->getUSinogramPixelNb() != uSinogramPixelNb)// sinogram_temp = new Sinogram3D_CPU_half(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb);
//
//
// //allocate device memory for sinogram result
// size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
// size_tuile= uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMalloc((void**) &sinogram_d, sizeof(half)*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()*nstreams));
//
//
// // //Copy constante
// // copy_constante_GPU_projection_ER(plan->constante_GPU_h,plan->sinogram_h->N_phi);
//
//
// //VOLUME MEMORY ALLOCATION
// //Create 3D array
// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
// struct cudaExtent volume_cu_array_size;
// cudaArray* volume_cu_array;
// volume_cu_array_size= make_cudaExtent(plan->volume_h->getXVolumePixelNb(),plan->volume_h->getYVolumePixelNb(),plan->volume_h->getZVolumePixelNb()/2+1);
// checkCudaErrors( cudaMalloc3DArray(&volume_cu_array, &channelDesc,volume_cu_array_size ) );
//
// // set texture parameters
// volume_tex.addressMode[0] = cudaAddressModeBorder;
// volume_tex.addressMode[1] = cudaAddressModeBorder;
// volume_tex.addressMode[2] = cudaAddressModeBorder;
// volume_tex.filterMode = cudaFilterModeLinear;
// volume_tex.normalized = false; // access with normalized texture coordinates
// // Bind the array to the 3D texture
// checkCudaErrors(cudaBindTextureToArray(volume_tex,volume_cu_array,channelDesc));
//
//
//
// //copy data to 3D array
// cudaMemcpy3DParms copyParams = {0};
// copyParams.dstArray = volume_cu_array;
// copyParams.extent = volume_cu_array_size;
// copyParams.kind = cudaMemcpyHostToDevice;
//
// if (gpuNb==1)
// {
// kv_start=0;
// kv_stop=kv_max/2;
//
// for(semi_plan_z=0; semi_plan_z<2;semi_plan_z++)
// {
// printf("kv_start=%d kv_stop=%d\n",kv_start,kv_stop);
// projection_ER_semi_volume_kv_start_kv_stop(semi_plan_z,kv_start,kv_stop,size_sinogram,size_tuile,plan,sinogram_d,sinogram_temp,&copyParams,streams,nstreams);
// kv_start+=kv_max/2;
// kv_stop+=kv_max/2;
// }
// }
// else
// {
// kv_start=(kv_max/gpuNb)*plan->device;
// kv_stop=(kv_max/gpuNb)*plan->device+kv_max/gpuNb;
// printf("GPU %d kv_start %d kv_stop %d\n",plan->device,kv_start,kv_stop);
// projection_ER_semi_volume_kv_start_kv_stop(plan->semi_plan_z,kv_start,kv_stop,size_sinogram,size_tuile,plan,sinogram_d,sinogram_temp,&copyParams,streams,nstreams);
// }
//
// checkCudaErrors(cudaFree(sinogram_d));
// checkCudaErrors(cudaFreeArray(volume_cu_array));
//
// if (plan->sinogram_h->getUSinogramPixelNb() != uSinogramPixelNb)// delete sinogram_temp;
//
// for(int i = 0 ; i < nstreams ; i++)
// {
// checkCudaErrors(cudaStreamDestroy(streams[i]));
//
// }
//
// CUT_THREADEND
//}
//
//void RegularSamplingProjector_CPU_half::projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj_half<Volume_CPU_half, Sinogram3D_CPU_half> *plan,half* sinogram_d,Sinogram3D_CPU_half *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams)
//{
// int un,vn,phi;
//
// int phi_start=0;
// int ni, k_phi;
// int N_un_start=0, N_vn_start=0, N_zn_start;
// int ku, kv, ku_max;
//
// unsigned long int uSinogramPixelNb = plan->cudaprojectionArchitecture->getXThreadNb()*plan->cudaprojectionArchitecture->getXBlockNb();
// unsigned long int vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
// unsigned long int projectionSinogramNb = plan->sinogram_h->getProjectionSinogramNb();
//
// dim3 dimBlock(plan->cudaprojectionArchitecture->getXThreadNb(),plan->cudaprojectionArchitecture->getYThreadNb(),plan->cudaprojectionArchitecture->getProjectionThreadNb());
// dim3 dimGrid(plan->cudaprojectionArchitecture->getXBlockNb(),plan->cudaprojectionArchitecture->getYBlockNb());
//
// //copy data to 3D array
// if(semi_plan_z==0)
// {
// N_zn_start=0;
// }
// else
// {
// N_zn_start=(plan->volume_h->getZVolumePixelNb()/2)-1;
// }
//
// copyParams->srcPtr = make_cudaPitchedPtr(plan->volume_h->getVolumeData()+semi_plan_z*(N_zn_start)*plan->volume_h->getXVolumePixelNb()*plan->volume_h->getYVolumePixelNb(), copyParams->extent.width*sizeof(half), copyParams->extent.width, copyParams->extent.height);// checkCudaErrors(cudaMemcpy3D(copyParams));
//
// // Decoupage des boucles de lancement du kernel
// k_phi = (int)(projectionSinogramNb/plan->cudaprojectionArchitecture->getProjectionThreadNb());
// ku_max=plan->sinogram_h->getUSinogramPixelNb()/uSinogramPixelNb;
//
// // EMPLACEMENT DES BOUCLES SUR LES GRILLES
// for(ku=0;ku<ku_max;ku++)
// {
// N_un_start=ku*(uSinogramPixelNb);
//
// for(kv=kv_start;kv<kv_stop;kv++)
// {
// N_vn_start=kv*(vSinogramPixelNb);
//
// printf("GPU %d N_vn_start %d\n",plan->device,N_vn_start);//
// for(int i = 0 ; i < nstreams ; i++)
// {
// phi_start=plan->cudaprojectionArchitecture->getProjectionThreadNb()*i;
// printf("GPU %d phi_start : %d\n",plan->device, phi_start);
// projection_ERB_kernel_v0_half<<< dimGrid, dimBlock,0,streams[i]>>>((unsigned short *)sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(),phi_start,N_un_start, N_vn_start, N_zn_start,uSinogramPixelNb,vSinogramPixelNb);
// }
//
// for(ni=0;ni<k_phi/nstreams-1;ni++)
// {
// for(int i = 0 ; i < nstreams ; i++)
// {
// phi_start=plan->cudaprojectionArchitecture->getProjectionThreadNb()*(ni*nstreams+i);
// printf("GPU %d phi_start : %d\n",plan->device, phi_start);
//
// if (plan->sinogram_h->getUSinogramPixelNb() == uSinogramPixelNb)
// {
// for (phi=0;phi<plan->cudaprojectionArchitecture->getProjectionThreadNb();phi++)
// {
// half* coord_sino_h;
// half* coord_sino_d;
// coord_sino_h=plan->sinogram_h->getDataSinogram()+N_un_start+N_vn_start*plan->sinogram_h->getUSinogramPixelNb()+(unsigned long int)(phi_start+phi)*(unsigned long int)plan->sinogram_h->getUSinogramPixelNb()*(unsigned long int)plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_d=sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()+phi*uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMemcpyAsync(coord_sino_h,coord_sino_d, size_tuile*sizeof(half),cudaMemcpyDeviceToHost,streams[i]) );
// }
// }
// else
// {
// checkCudaErrors(cudaMemcpyAsync(sinogram_temp->getDataSinogram(),sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(), size_sinogram*sizeof(half),cudaMemcpyDeviceToHost,streams[i]));
//
// for (phi=0;phi<k_phi*plan->cudaprojectionArchitecture->getProjectionThreadNb();phi++)
// for (vn=0;vn<sinogram_temp->getVSinogramPixelNb();vn++)
// for (un=0;un<sinogram_temp->getUSinogramPixelNb();un++)
// {
// unsigned long int coord_sino;
// unsigned long int coord_sino_temp;
// coord_sino=(un+N_un_start)+(vn+N_vn_start)*plan->sinogram_h->getUSinogramPixelNb()+(phi+plan->phi_start)*plan->sinogram_h->getUSinogramPixelNb()*plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_temp=un+vn*sinogram_temp->getUSinogramPixelNb()+(phi+plan->phi_start)*sinogram_temp->getUSinogramPixelNb()*sinogram_temp->getVSinogramPixelNb();
// plan->sinogram_h->getDataSinogram()[coord_sino]= sinogram_temp->getDataSinogram()[coord_sino_temp];
// }
// }
// phi_start=plan->cudaprojectionArchitecture->getProjectionThreadNb()*((ni+1)*nstreams+i);
// projection_ERB_kernel_v0_half<<< dimGrid, dimBlock,0,streams[i]>>>((unsigned short *)sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(),phi_start,N_un_start, N_vn_start, N_zn_start,uSinogramPixelNb,vSinogramPixelNb);
// }//loop sur i nstreams
// }//loop sur kphi
//
// for(int i = 0 ; i < nstreams ; i++)
// {
// phi_start=plan->cudaprojectionArchitecture->getProjectionThreadNb()*(ni*nstreams+i);
//
// if (plan->sinogram_h->getUSinogramPixelNb() == uSinogramPixelNb)
// {
// for (phi=0;phi<plan->cudaprojectionArchitecture->getProjectionThreadNb();phi++)
// {
// half *coord_sino_h;
// half *coord_sino_d;
// coord_sino_h=plan->sinogram_h->getDataSinogram()+N_un_start+N_vn_start*plan->sinogram_h->getUSinogramPixelNb()+(unsigned long int)(phi_start+phi)*(unsigned long int)plan->sinogram_h->getUSinogramPixelNb()*(unsigned long int)plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_d=sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()+phi*uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMemcpyAsync(coord_sino_h,coord_sino_d, size_tuile*sizeof(half),cudaMemcpyDeviceToHost,streams[i]) );
// }
// }
// else
// {
// checkCudaErrors(cudaMemcpyAsync(sinogram_temp->getDataSinogram(), sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(), size_sinogram*sizeof(half),cudaMemcpyDeviceToHost,streams[i]) );
//
// for (phi=0;phi<k_phi*plan->cudaprojectionArchitecture->getProjectionThreadNb();phi++)
// for (vn=0;vn<sinogram_temp->getVSinogramPixelNb();vn++)
// for (un=0;un<sinogram_temp->getUSinogramPixelNb();un++)
// {
// unsigned long int coord_sino;
// unsigned long int coord_sino_temp;
// coord_sino=(un+N_un_start)+(vn+N_vn_start)*plan->sinogram_h->getUSinogramPixelNb()+(phi+plan->phi_start)*plan->sinogram_h->getUSinogramPixelNb()*plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_temp=un+vn*sinogram_temp->getUSinogramPixelNb()+(phi+plan->phi_start)*sinogram_temp->getUSinogramPixelNb()*sinogram_temp->getVSinogramPixelNb();
// plan->sinogram_h->getDataSinogram()[coord_sino]= sinogram_temp->getDataSinogram()[coord_sino_temp];
// }
// }
// }
// }
// }
//}


/* RegularSamplingProjector definition */
RegularSamplingProjector_GPU_half::RegularSamplingProjector_GPU_half() : Projector_half<Volume_GPU_half,Sinogram3D_GPU_half>(){}


/* RegularSamplingProjector definition */
RegularSamplingProjector_GPU_half::RegularSamplingProjector_GPU_half(Acquisition* acquisition, Detector* detector, CUDAProjectionArchitecture*  cudaprojectionArchitecture,Volume_GPU_half* volume) : Projector_half<Volume_GPU_half,Sinogram3D_GPU_half>(acquisition, detector,  cudaprojectionArchitecture,volume){

	cout << "********** Start Constant Copy **********" << endl;
	cout << "Projection Constant Copy on device n° " << 0 << endl;
	checkCudaErrors(cudaSetDevice(0));
	this->copyConstantGPU();
	cout << "********** End Projection Constant Copy **********" << endl;}
RegularSamplingProjector_GPU_half::~RegularSamplingProjector_GPU_half(){}

void RegularSamplingProjector_GPU_half::doProjection(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume)
{


	std::cout << "Regular Sampling Projection all on GPU" << std::endl;

	unsigned long long int xVolumePixelNb = volume->getXVolumePixelNb(); unsigned long long int yVolumePixelNb = volume->getYVolumePixelNb();
	unsigned long long int zVolumePixelNb = volume->getZVolumePixelNb();

	std::cout << xVolumePixelNb << yVolumePixelNb << zVolumePixelNb << std::endl;

	checkCudaErrors(cudaSetDevice(0));

	dim3 dimBlock(this->getCUDAProjectionArchitecture()->getXThreadNb(),this->getCUDAProjectionArchitecture()->getYThreadNb(),this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	dim3 dimGrid(estimatedSinogram->getUSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getXThreadNb(),estimatedSinogram->getVSinogramPixelNb()/this->getCUDAProjectionArchitecture()->getYThreadNb(),estimatedSinogram->getProjectionSinogramNb()/this->getCUDAProjectionArchitecture()->getProjectionThreadNb());


	//VOLUME MEMORY ALLOCATION
	//Create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(half), 0, 0, 0, cudaChannelFormatKindFloat);
	struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	volume_cu_array_size= make_cudaExtent(xVolumePixelNb,yVolumePixelNb,zVolumePixelNb);
	checkCudaErrors(cudaMalloc3DArray(&volume_cu_array, &channelDesc,volume_cu_array_size ) );

	//copy data to 3D array
	cudaMemcpy3DParms VolumeParams ={0};
	VolumeParams.srcPos = make_cudaPos(0,0,0);
	VolumeParams.dstPos = make_cudaPos(0,0,0);
	VolumeParams.dstArray = volume_cu_array;
	VolumeParams.extent = volume_cu_array_size;
	VolumeParams.kind = cudaMemcpyDeviceToDevice;

	half* volumeData = volume->getVolumeData();
	half* estimatedSinogramData = estimatedSinogram->getDataSinogram();
	VolumeParams.srcPtr = make_cudaPitchedPtr(volumeData, VolumeParams.extent.width*sizeof(half), VolumeParams.extent.width, VolumeParams.extent.height);

	// set texture parameters
	volume_tex.addressMode[0] = cudaAddressModeBorder;
	volume_tex.addressMode[1] = cudaAddressModeBorder;
	volume_tex.addressMode[2] = cudaAddressModeBorder;
	volume_tex.filterMode = cudaFilterModeLinear;
	volume_tex.normalized = false; // access with normalized texture coordinates
	// Bind the array to the 3D texture
	checkCudaErrors(cudaBindTextureToArray(volume_tex, volume_cu_array, channelDesc));
	checkCudaErrors(cudaMemcpy3D(&VolumeParams));
	projection_ERB_kernel_v0_half_UM<<< dimGrid, dimBlock>>>((estimatedSinogram->getDataSinogram()));
	cudaDeviceSynchronize();

	checkCudaErrors(cudaFreeArray(volume_cu_array));
}

/*
void RegularSamplingProjector_GPU_half::doProjectionSFTR(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume){}

void RegularSamplingProjector_GPU_half::doProjectionSFTR_2kernels(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume){}

void RegularSamplingProjector_GPU_half::doProjectionSFTR_opti(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume){}

void RegularSamplingProjector_GPU_half::weightedCoeffDiagHVHTSFTR(Sinogram3D_GPU_half* coeffDiag,Volume_GPU_half* weights){}

void RegularSamplingProjector_GPU_half::weightedCoeffDiagHVHTSFTR_2kernels(Sinogram3D_GPU_half* coeffDiag,Volume_GPU_half* weights){}

void RegularSamplingProjector_GPU_half::doProjectionSFTR_allCPU(Sinogram3D_GPU_half* estimatedSinogram,Volume_GPU_half *volume){}
*/

///* JosephProjector definition */
//template <template<typename> class V, template<typename> class S,typename T>
//JosephProjector<V,S,T>::JosephProjector(Acquisition* acquisition, Detector* detector, V<T>* volume) : Projector<V,S,T>(acquisition, detector, volume)
//{
// float startAngle = acquisition->getStartAngle();
// float focusObjectDistance = acquisition->getFocusObjectDistance();// float focusDetectorDistance = acquisition->getFocusDetectorDistance();
// float xVolumePixelSize = volume->getXVolumePixelSize();
// float zVolumePixelSize = volume->getZVolumePixelSize();
// float uDetectorPixelSize = detector->getUDetectorPixelSize();
// float vDetectorPixelSize = detector->getVDetectorPixelSize();
//
//
// unsigned short projectionNb = acquisition->getProjectionNb();
// this->alphaPreComputingC = new float[projectionNb];
// this->betaPreComputingC = new float[projectionNb];
// this->deltaPreComputingC = new float[projectionNb];
// this->sigmaPreComputingC = new float[projectionNb];
// this->kappaPreComputingC = new float[projectionNb];
// this->iotaPreComputingC = new float[projectionNb];
//
// gammaPrecomputingC = focusObjectDistance/xVolumePixelSize;
// omegaPrecomputingC = (vDetectorPixelSize*focusObjectDistance)/(focusDetectorDistance/zVolumePixelSize);
//
// double* phiValueTab = acquisition->getPhiValue();
//
// for (int p=0;p<projectionNb;p++)
// {
// alphaPreComputingC[p] = cos(phiValueTab[p]);
// betaPreComputingC[p] = (focusDetectorDistance/uDetectorPixelSize)*sin(phiValueTab[p]);
// deltaPreComputingC[p] = -1.0*sin(phiValueTab[p]);
// sigmaPreComputingC[p] = (focusDetectorDistance/uDetectorPixelSize)*cos(phiValueTab[p]);
// kappaPreComputingC[p] = (vDetectorPixelSize*xVolumePixelSize*sin(phiValueTab[p]))/(focusDetectorDistance*zVolumePixelSize);
// iotaPreComputingC[p] = (vDetectorPixelSize*xVolumePixelSize*cos(phiValueTab[p]))/(focusDetectorDistance*zVolumePixelSize);
// }
//}
//
//template <template<typename> class V, template<typename> class S,typename T>
//JosephProjector<V,S,T>::~JosephProjector()
//{
// delete alphaPreComputingC;
// delete betaPreComputingC;
// delete deltaPreComputingC;
// delete sigmaPreComputingC;
// delete kappaPreComputingC;
// delete iotaPreComputingC;
//}
//
//template <template<typename> class V, template<typename> class S,typename T>
//void JosephProjector<V,S,T>::doProjection_GPU(S<T>* estimatedSinogram, CUDAProjectionArchitecture*  cudaprojectionArchitecture)
//{
// std::cout << "Joseph Projection" << std::endl;
//}
//
//template <template<typename> class V, template<typename> class S,typename T>
//void JosephProjector<V,S,T>::doProjection_CPU_GPU(S<T>* estimatedSinogram, CUDAProjectionArchitecture*  cudaprojectionArchitecture)
//{
// std::cout << "Joseph Projection" << std::endl;
//}
//template class JosephProjector<Volume_CPU, Sinogram3D_CPU,int>; // 16-bit signed image
//template class JosephProjector<Volume_CPU, Sinogram3D_CPU,short>; // 16-bit signed image
//template class JosephProjector<Volume_CPU, Sinogram3D_CPU,float>; // 32-bit unsigned image
//template class JosephProjector<Volume_CPU, Sinogram3D_CPU,double>; // 64-bit signed image
//template class JosephProjector<Volume_GPU, Sinogram3D_GPU,float>; // 32-bit unsigned image
//template class JosephProjector<Volume_GPU, Sinogram3D_GPU,double>; // 64-bit signed image
////template class JosephProjector<Volume_GPU_half, Sinogram3D_GPU_half,half>; // 64-bit signed image

#include "Projector_instances_CPU.cu"
#include "Projector_instances_GPU.cu"