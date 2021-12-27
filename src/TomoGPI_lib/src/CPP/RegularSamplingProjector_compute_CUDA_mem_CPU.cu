/*
 *
  * Author: gac
 */

#include "Projector_CPU.cuh"
#include "Projector_kernel.cuh"

/* RegularSamplingProjector definition */
/*template <typename T>
RegularSamplingProjector_CPU<T>::RegularSamplingProjector_CPU() : Projector<Volume_CPU,Sinogram3D_CPU,T>() {}*/

/* RegularSamplingProjector definition */
template <typename T>
RegularSamplingProjector_compute_CUDA_mem_CPU<T>::RegularSamplingProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector,CUDAProjectionArchitecture*  cudaprojectionArchitecture, Volume_CPU<T>* volume) : Projector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector, volume){

this->setCUDAProjectionArchitecture(cudaprojectionArchitecture);


}

template <typename T>
RegularSamplingProjector_compute_CUDA_mem_CPU<T>::~RegularSamplingProjector_compute_CUDA_mem_CPU(){}


template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume)
{
	//std::cout << "\tRegular Sampling Projection running on CPU " << sched_getcpu() << std::endl;
	//std::cout << "\tAdress sinogram " << estimatedSinogram->getDataSinogram() << std::endl;

	std::cout << "\tRegular Sampling Projection on CPU" << std::endl;
	this->setVolume(volume);

	TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T>* plan;
	std::thread **threadID;
	cpu_set_t cpuset,cpuset_get;

	this->setVolume(volume);

	//counter=0;

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
	taille_allocation_vol=(size_t)sizeof(T)*(size_t)xVolumePixelNb*yVolumePixelNb*(size_t)zVolumePixelNb;//il faut allouer 1 volume
	taille_allocation_sino=(size_t)sizeof(T)*size_sinogram;//*(size_t)nstreams;//il faut allouer 1 volume
	taille_allocation=taille_allocation_vol+taille_allocation_sino;
	ratio_allocation_SDRAM=taille_allocation/taille_SDRAM;
	//printf("allocation : %.2f Go (vol %.2f Go sino %.2f Go) SDRAM : %.2f Go ratio :%.2f\n",taille_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_vol/gpuNb)/((1024.0*1024.0*1024.0)),taille_allocation_sino/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_allocation_SDRAM);



	//printf("nb_blocs_v_par_device %d ",nb_bloc_v_par_device);
	taille_bloc_allocation=taille_allocation_vol/((float)nb_bloc_v_par_device*(float)gpuNb)+taille_allocation_sino;
	ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;

	int nb2 = this->getCUDAProjectionArchitecture()->getYBlockNb();
	//printf("%d \n",nb2);
	while(taille_allocation_sino/taille_SDRAM>=0.5)
	{
		if (nb2>1)
		{
			nb2/=2;//printf("%d \n",nb2);
			this->getCUDAProjectionArchitecture()->setYBlockNb(nb2);
		}
		else
			nstreams/=2;

		vSinogramPixelNb = this->getCUDAProjectionArchitecture()->getYThreadNb()*this->getCUDAProjectionArchitecture()->getYBlockNb();

		size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
		taille_allocation_sino=(size_t)sizeof(T)*size_sinogram*(size_t)nstreams;

	}



	while(ratio_bloc_SDRAM>=0.7)
	{
		nb_bloc_v_par_device*=2;
		//printf("%d \n",nb_bloc_v_par_device);
		taille_bloc_allocation=taille_allocation_vol/((float)nb_bloc_v_par_device*(float)gpuNb)+taille_allocation_sino;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	}

	//printf("\n");
	//printf("allocation par bloc : %.2f Go (vol %.2f Go sino %.2f Go) SDRAM : %.2f Go ratio :%.2f\n",taille_bloc_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_vol/(nb_bloc_v_par_device*gpuNb))/((1024.0*1024.0*1024.0)),taille_allocation_sino/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_bloc_SDRAM);

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
	N_vn_par_kernel=16;//this->getCUDAProjectionArchitecture()->getYBlockNb()
	N_ligne_par_carte=(int)(N_vn_par_solverthread/N_vn_par_kernel);
	//	printf("N_ligne_par_carte %d N_vn_par_carte %d N_vn_par_solverthread %d nb_bloc_v_par_device %d \n",N_ligne_par_carte,N_vn_par_carte,N_vn_par_solverthread,nb_bloc_v_par_device);

	while (N_ligne_par_carte%nstreams!=0)
	{
		if (this->getCUDAProjectionArchitecture()->getYBlockNb()>1)
			this->getCUDAProjectionArchitecture()->setYBlockNb(this->getCUDAProjectionArchitecture()->getYBlockNb()/2);
		else
			nstreams/=2;
		//N_vn_par_kernel=16this->getCUDAProjectionArchitecture()->getYBlockNb();
		N_ligne_par_carte=(int)(N_vn_par_solverthread/N_vn_par_kernel);
	}
	this->getCUDAProjectionArchitecture()->setYBlockNb(N_vn_par_kernel);
	this->getCUDAProjectionArchitecture()->setProjectionStreamsNb(nstreams);


	//printf("N_ligne_par_carte %d N_vn_par_carte %d N_vn_par_solverthread %d nb_bloc_v_par_device %d \n",N_ligne_par_carte,N_vn_par_carte,N_vn_par_solverthread,nb_bloc_v_par_device);
	cudaEvent_t *start_thread;
	cudaEvent_t *stop_thread;

	start_thread=(cudaEvent_t*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(cudaEvent_t));
	stop_thread=(cudaEvent_t*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(cudaEvent_t));

	plan=(TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T>*)malloc(gpuNb*nb_bloc_v_par_device*sizeof(TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T>));
	//threadID=(CUTThread *)malloc(gpuNb*nb_bloc_v_par_device*sizeof(CUTThread));
	threadID=(std::thread **)malloc(gpuNb*sizeof(std::thread *));


	uSinogramPixelNb = this->getCUDAProjectionArchitecture()->getXThreadNb()*this->getCUDAProjectionArchitecture()->getXBlockNb();
	vSinogramPixelNb = this->getCUDAProjectionArchitecture()->getYThreadNb()*this->getCUDAProjectionArchitecture()->getYBlockNb();
	projectionSinogramNb = this->getCUDAProjectionArchitecture()->getProjectionThreadNb()*this->getCUDAProjectionArchitecture()->getZBlockNb();

	dim3 dimBlock(this->getCUDAProjectionArchitecture()->getXThreadNb(),this->getCUDAProjectionArchitecture()->getYThreadNb(),this->getCUDAProjectionArchitecture()->getProjectionThreadNb());
	dim3 dimGrid(this->getCUDAProjectionArchitecture()->getXBlockNb(), this->getCUDAProjectionArchitecture()->getYBlockNb(), this->getCUDAProjectionArchitecture()->getZBlockNb());

	int N_vn_restant;
	N_vn_par_kernel=this->getCUDAProjectionArchitecture()->getYBlockNb()*this->getCUDAProjectionArchitecture()->getYThreadNb();
	N_ligne_par_carte=(int)(N_vn_par_solverthread/N_vn_par_kernel);

	N_vn_restant=((float)(N_vn_par_solverthread)/N_vn_par_kernel)-N_ligne_par_carte;
	if(N_vn_restant>0)
	{
		N_ligne_par_carte+=1;
	}


	for(int device=0;device<gpuNb;device++)
	{
		for (int n=0;n<nb_bloc_v_par_device;n++){
			if (device%2==1)
				num_bloc[n+device*nb_bloc_v_par_device]=(nb_bloc_v_par_device-1)-n;
			else
				num_bloc[n+device*nb_bloc_v_par_device]=n;

			//printf("n %d device %d num_device %d n+device*nb_bloc_v_par_device %d num_bloc %d\n",n,device,num_device[device],n+device*nb_bloc_v_par_device,num_bloc[n+device*nb_bloc_v_par_device]);
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


			plan[n+device*nb_bloc_v_par_device].nstreams=this->getCUDAProjectionArchitecture()->getProjectionStreamsNb();

			//allocate device memory for sinogram result
			plan[n+device*nb_bloc_v_par_device].size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
			plan[n+device*nb_bloc_v_par_device].dimBlock=dimBlock;
			plan[n+device*nb_bloc_v_par_device].dimGrid=dimGrid;

			plan[n+device*nb_bloc_v_par_device].N_vn_par_kernel=N_vn_par_kernel;
			plan[n+device*nb_bloc_v_par_device].N_ligne_par_carte=N_ligne_par_carte;
			plan[n+device*nb_bloc_v_par_device].N_vn_restant=N_vn_restant;
			//cout << "size of volume h : "<< sizeof(plan[n+device*nb_bloc_v_par_device].volume_h)<<"\t" << this->getVolume() << endl;
			//cout << "size of volume : "<< sizeof(volume)<<"\t" << volume << endl;
			//cout << "nb : "<< nb_bloc_v_par_device <<" n "<< n <<" devce "<<device<<"  "<< n+device*nb_bloc_v_par_device << endl;


		}
	}

	for(int device=0;device<gpuNb;device++)
	{
		num_device[device]=device;

		//cout << "********** Start Constant Copy **********" << endl;
		//cout << "Projection Constant Copy on device n° " << device << endl;
		cudaSetDevice(device);
		this->copyConstantGPU();
		//cout << "********** End Projection Constant Copy **********" << endl;
	}



	for (int n=0;n<nb_bloc_v_par_device;n++){
		for(int device=0;device<gpuNb;device++){
			int vn_start,vn_stop;
			float vn_prime_start,vn_prime_stop;
			int zn_start,zn_stop;
			float zn_prime_start,zn_prime_stop;
			float zn_start_f,zn_stop_f;

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

			zn_stop_f=zn_prime_stop+zVolumeCenterPixel;
			zn_stop=ceil(zn_stop_f);

			if (zn_stop>=zVolumePixelNb)
				zn_stop=zVolumePixelNb-1;

			zn_start_f=zn_prime_start+zVolumeCenterPixel;
			zn_start=floor(zn_start_f);

			if (zn_start<0)
				zn_start=0;

			//printf("device %d bloc %d vn_start %d vn_prime_start %f vn_prime_stop %f zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_solverthread %d\n",device,n,vn_start,vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1);

			plan[n+device*nb_bloc_v_par_device].vn_start=vn_start;
			plan[n+device*nb_bloc_v_par_device].zn_start=zn_start;
			plan[n+device*nb_bloc_v_par_device].N_zn_par_solverthread=(zn_stop-zn_start)+1;

			//CPU_ZERO(&cpuset);
			//CPU_SET(2*8-2-device, &cpuset);
			//CPU_SET(0, &cpuset);
			/*for (int i=(device/4)*8;i<(device/4)*8+8;i++){
//printf("device=%d i=%d\n",device,i);
CPU_SET(i, &cpuset);
}*/
			/*for (int i=0;i<8;i++){
//printf("device=%d i=%d\n",device,i);
CPU_SET(i, &cpuset);
}*/
			//CPU_SET(device, &cpuset);
			//threadID[device+n*nb_bloc_v_par_device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_v_par_device));

			threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_v_par_device));
			// counter++;



			/*int rc = pthread_setaffinity_np(threadID[device]->native_handle(),sizeof(cpu_set_t), &cpuset);
if (rc != 0) {
printf("ERROR: %s\n", strerror(errno));
std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
}*/

			//std::cout << "Joinable after construction:\n" << std::boolalpha;
			// std::cout << device << " : " << threadID[device]->joinable() << '\n';

		}



		//cutWaitForThreads(threadID+n*nb_bloc_v_par_device,gpuNb);
		for (int i = 0; i < gpuNb; i++)
		{
			/*while (1)
{
// Use a lexical scope and lock_guard to safely lock the mutex only
// for the duration of std::cout usage.
std::lock_guard<std::mutex> iolock(iomutex);
std::cout << "Thread #" << i << ": on CPU " << sched_getcpu() << "\n";
}*/

			if (threadID[i]->joinable()){
				// std::cout << i <<" joined:\n" << endl;
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


template <typename T>
CUT_THREADPROC RegularSamplingProjector_compute_CUDA_mem_CPU<T>::solverThread(TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T> *plan)
{

	float vn_prime_start,vn_prime_stop;
	int *vn_start;//,*vn_start_old;
	int zn_start,zn_stop,zn_start_old,zn_stop_old;

	int kligne;
	float zn_prime_start,zn_prime_stop;
	float zn_start_f,zn_stop_f;

	float fdd = plan->acquisition->getFocusDetectorDistance();
	float fod = plan->acquisition->getFocusObjectDistance();
	float vDetectorPixelSize = plan->detector->getVDetectorPixelSize();
	float vDetectorCenterPixel = plan->detector->getVDetectorCenterPixel();

	float xVolumePixelSize = plan->volume_h->getXVolumePixelSize();
	float zVolumeCenterPixel = plan->volume_h->getZVolumeCenterPixel();
	unsigned long int xVolumePixelNb = plan->volume_h->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = plan->volume_h->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = plan->volume_h->getZVolumePixelNb();

	unsigned long long int uSinogramPixelNb = plan->cudaprojectionArchitecture->getXThreadNb()*plan->cudaprojectionArchitecture->getXBlockNb();
	unsigned long long int vSinogramPixelNb = plan->cudaprojectionArchitecture->getYThreadNb()*plan->cudaprojectionArchitecture->getYBlockNb();
	unsigned long long int projectionSinogramNb = plan->cudaprojectionArchitecture->getProjectionThreadNb()*plan->cudaprojectionArchitecture->getZBlockNb();

	unsigned long int uSinoPixelNb = plan->sinogram_h->getUSinogramPixelNb();
	unsigned long int vSinoPixelNb = plan->sinogram_h->getVSinogramPixelNb();
	unsigned long int phiSinoNb = plan->sinogram_h->getProjectionSinogramNb();


	vn_start=(int*)malloc(plan->nstreams*sizeof(int));
	//vn_start_old=(int*)malloc(plan->nstreams*sizeof(int));

	T* volumeData = plan->volume_h->getVolumeData();
	T* estimatedSinogramData = plan->sinogram_h->getDataSinogram();

	T** sinogram_df;
	volatile cudaStream_t *streams;


	// Use a lexical scope and lock_guard to safely lock the mutex only // for the duration of std::cout usage.
	// iomutex.lock();
	/*char pciBusId[255];
checkCudaErrors(cudaDeviceGetPCIBusId(pciBusId, 255,plan->device));
std::cout << "Device #" << plan->device << "bus id" << pciBusId << ": on CPU " << sched_getcpu() << "\n";*/

	// iomutex.unlock();



	//Set device
	checkCudaErrors(cudaSetDevice(plan->device));
	//printf("GPU : %d \n",plan->device);

	cudaEvent_t start_solverthread,stop_solverthread;
	checkCudaErrors(cudaEventCreate(&start_solverthread));
	checkCudaErrors(cudaEventCreate(&stop_solverthread));
	checkCudaErrors(cudaEventRecord(start_solverthread, NULL));


	streams = (cudaStream_t*) malloc((plan->nstreams)*sizeof(cudaStream_t));
	for(int i=0; i<plan->nstreams ; i++)
		checkCudaErrors(cudaStreamCreate((cudaStream_t *)&streams[i])) ;

	volatile cudaEvent_t *event;
	event=(cudaEvent_t *)malloc(plan->nstreams*sizeof(cudaEvent_t)); for(int i=0;i<plan->nstreams;i++)
		checkCudaErrors(cudaEventCreate((cudaEvent_t *)event+i));



	sinogram_df=(T**)malloc(sizeof(T*)*plan->nstreams);
	for(int i=0;i<plan->nstreams;i++)
		checkCudaErrors(cudaMalloc((void**) &(sinogram_df[i]), (unsigned long long int)sizeof(T)*(unsigned long long int)plan->size_sinogram));

	//VOLUME MEMORY ALLOCATION
	//Create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(T), 0, 0, 0, cudaChannelFormatKindFloat);
	struct cudaExtent volume_cu_array_size;
	cudaArray* volume_cu_array;
	volume_cu_array_size= make_cudaExtent((size_t)plan->volume_h->getXVolumePixelNb(),(size_t)plan->volume_h->getYVolumePixelNb(),(size_t)plan->N_zn_par_solverthread);
	checkCudaErrors( cudaMalloc3DArray(&volume_cu_array, &channelDesc,volume_cu_array_size ) );

	cudaMemcpy3DParms copyParams={0} ;

	copyParams.dstArray = volume_cu_array;
	copyParams.kind = cudaMemcpyHostToDevice;

	// set texture parameters
	volume_tex.addressMode[0] = cudaAddressModeBorder;
	volume_tex.addressMode[1] = cudaAddressModeBorder;
	volume_tex.addressMode[2] = cudaAddressModeBorder;
	volume_tex.filterMode = cudaFilterModeLinear;
	volume_tex.normalized = false; // access with normalized texture coordinates
	// Bind the array to the 3D texture
	checkCudaErrors(cudaBindTextureToArray(volume_tex,volume_cu_array,channelDesc));

	/*iomutex.lock();
counter--;
iomutex.unlock();
while(counter!=0);*/

	//printf("Je passe par la !\n");

	for(kligne=0;kligne<((plan->N_ligne_par_carte/plan->nstreams));kligne++)
	{
		int i;
		i=0;


		zn_stop_old=zn_stop;
		vn_start[i]=plan->vn_start+(i+kligne*plan->nstreams)*plan->N_vn_par_kernel;
		vn_prime_start=vn_start[i]-vDetectorCenterPixel;
		vn_prime_stop=vn_prime_start+plan->N_vn_par_kernel;

		if(vn_prime_stop>=0)
			zn_prime_stop=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
		else
			zn_prime_stop=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/((float)fdd*xVolumePixelSize);

		zn_stop_f=zn_prime_stop+zVolumeCenterPixel;
		zn_stop=ceil(zn_stop_f);

		if (zn_stop>=zVolumePixelNb)
			zn_stop=zVolumePixelNb-1;

		if (kligne!=0)
			zn_start=zn_stop_old+1;
		else {
			if(vn_prime_start>=0)
				zn_prime_start=((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);
			else
				zn_prime_start=((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_start*vDetectorPixelSize/((float)fdd*xVolumePixelSize);

			zn_start_f=zn_prime_start+zVolumeCenterPixel;
			zn_start=floor(zn_start_f);

			if (zn_start<0)
				zn_start=0;
			if ((zn_start-plan->zn_start)<0)
				zn_start=plan->zn_start;
		}
		//printf("device %d vn_start %d vn_prime_start %f vn_prime_stop %f zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_kernel %d\n",plan->device,vn_start[i],vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1);



		copyParams.srcPos = make_cudaPos(0,0,0);//make_cudaPos(0,plan->vn_start,phi);
		copyParams.dstPos = make_cudaPos(0,0,(size_t)(zn_start-plan->zn_start));

		//printf("zn_start-plan->zn_start %d (zn_stop-zn_start)+1 %d\n",zn_start-plan->zn_start,(zn_stop-zn_start)+1);
		size_t coord_d;
		coord_d=(size_t)zn_start*(size_t)xVolumePixelNb*(size_t)yVolumePixelNb;

		copyParams.srcPtr = make_cudaPitchedPtr(&volumeData[coord_d],(size_t)xVolumePixelNb*(size_t)sizeof(T),(size_t)xVolumePixelNb,(size_t)yVolumePixelNb);

		copyParams.extent = make_cudaExtent((size_t)xVolumePixelNb,(size_t)yVolumePixelNb,(size_t)((zn_stop-zn_start)+1));

		if (((zn_stop-zn_start)+1)>0){

			checkCudaErrors(cudaMemcpy3DAsync(&copyParams,streams[i]));
		}


		for(i=1; i < plan->nstreams ; i++)
		{

			zn_stop_old=zn_stop;
			//vn_start_old[i]=vn_start[i-1];
			vn_start[i]=plan->vn_start+(i+kligne*plan->nstreams)*plan->N_vn_par_kernel;
			vn_prime_start=vn_start[i]-vDetectorCenterPixel;
			vn_prime_stop=vn_prime_start+plan->N_vn_par_kernel;

			if(vn_prime_stop>=0)
				zn_prime_stop=(fod+xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/(fdd*xVolumePixelSize);
			else
				zn_prime_stop=(fod-xVolumePixelNb*xVolumePixelSize/2.0)*(float)vn_prime_stop*vDetectorPixelSize/(fdd*xVolumePixelSize);

			zn_stop_f=zn_prime_stop+zVolumeCenterPixel;
			zn_stop=ceil(zn_stop_f);

			if (zn_stop>=zVolumePixelNb)
				zn_stop=zVolumePixelNb-1;

			zn_start=zn_stop_old+1;



			//checkCudaErrors(cudaStreamWaitEvent( streams[i], event[i-1],0 ));

			//printf("device %d vn_start %d vn_start_old %d vn_prime_start %f vn_prime_stop %f zn_prime_start %f zn_prime_stop %f zn_start %d zn_stop %d N_zn_par_kernel %d\n",plan->device,vn_start[i],vn_start_old,vn_prime_start,vn_prime_stop,zn_prime_start,zn_prime_stop,zn_start,zn_stop,(zn_stop-zn_start)+1);

			copyParams.srcPos = make_cudaPos(0,0,0);//make_cudaPos(0,plan->vn_start,phi);
			copyParams.dstPos = make_cudaPos(0,0,zn_start-plan->zn_start);

			//printf("zn_start-plan->zn_start %d (zn_stop-zn_start)+1 %d\n",zn_start-plan->zn_start,(zn_stop-zn_start)+1);
			size_t coord_d;
			coord_d=(size_t)zn_start*(size_t)xVolumePixelNb*(size_t)yVolumePixelNb;

			copyParams.srcPtr = make_cudaPitchedPtr(&(volumeData[coord_d]), xVolumePixelNb*sizeof(T), xVolumePixelNb, yVolumePixelNb);

			copyParams.extent = make_cudaExtent((unsigned long long int)xVolumePixelNb,(unsigned long long int)yVolumePixelNb,(unsigned long long int)(zn_stop-zn_start)+1);




			if (((zn_stop-zn_start)+1)>0){
				checkCudaErrors(cudaMemcpy3DAsync(&copyParams,streams[i]));
			}

			// checkCudaErrors(cudaEventRecord (event[i], streams[i]));


		}
		i=0;

		//printf("Kernel Launch");
		//printf("TX : %d, TY : %d, TZ : %d", plan->dimBlock.x, plan->dimBlock.y, plan->dimBlock.z);
		projection_ERB_kernel_v1<<< plan->dimGrid, plan->dimBlock,0,streams[i]>>>((T*)(sinogram_df[i]),vn_start[i],plan->zn_start);

		for(i=1; i < plan->nstreams ; i++)
		{
			projection_ERB_kernel_v1<<< plan->dimGrid, plan->dimBlock,0,streams[i]>>>((T*)(sinogram_df[i]),vn_start[i],plan->zn_start);
		}

		i=0;
		for (int phi=0;phi<phiSinoNb;phi++){
			unsigned long long int coord_sino_h;
			unsigned long long int coord_sino_d;
			coord_sino_h=(unsigned long int)(vn_start[0]*uSinoPixelNb)+(unsigned long int)(phi)*(unsigned long int)(uSinoPixelNb)*(unsigned long int)(vSinoPixelNb);
			coord_sino_d=(unsigned long int)(phi)*(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb;
			checkCudaErrors(cudaMemcpyAsync(estimatedSinogramData+coord_sino_h,sinogram_df[i]+coord_sino_d,(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb*sizeof(T),cudaMemcpyDeviceToHost,streams[i]));
		}

		for(i=1; i < plan->nstreams ; i++)
		{

			for (int phi=0;phi<phiSinoNb;phi++)
			{
				unsigned long long int coord_sino_h;
				unsigned long long int coord_sino_d;
				coord_sino_h=(unsigned long int)(vn_start[i]*uSinoPixelNb)+(unsigned long int)(phi)*(unsigned long int)(uSinoPixelNb)*(unsigned long int)(vSinoPixelNb);
				coord_sino_d=(unsigned long int)(phi)*(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb;
				checkCudaErrors(cudaMemcpyAsync(estimatedSinogramData+coord_sino_h,sinogram_df[i]+coord_sino_d,(unsigned long int)uSinogramPixelNb*(unsigned long int)vSinogramPixelNb*sizeof(T),cudaMemcpyDeviceToHost,streams[i]));
			}

		}


	}



	for(int i=0 ; i < plan->nstreams ; i++)
	{
		cudaStreamSynchronize(streams[i]);
		cudaEventDestroy(event[i]);
	}

	for(int i=0;i<plan->nstreams;i++)
		checkCudaErrors(cudaFree(sinogram_df[i]));

	free(sinogram_df);
	checkCudaErrors(cudaFreeArray(volume_cu_array));


	for(int i = 0 ; i < plan->nstreams ; i++)
	{
		checkCudaErrors(cudaStreamDestroy(streams[i]));

	}

	free((cudaStream_t*)streams);


	checkCudaErrors(cudaEventRecord(stop_solverthread, NULL));
	checkCudaErrors(cudaEventSynchronize(stop_solverthread));

	cudaEventDestroy(start_solverthread);
	cudaEventDestroy(stop_solverthread);

	CUT_THREADEND;


}




//template <typename T>
//CUT_THREADPROC RegularSamplingProjector_CPU<T>::solverThread(TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T> *plan)
//{
// unsigned long int size_sinogram;
// unsigned long int size_tuile;
//
// unsigned int kv_start,kv_stop,kv_max;
// unsigned int semi_plan_z;
// float nb_kv_par_device;
// Sinogram3D_CPU<T>* sinogram_temp;
// int gpuNb = plan->cudaprojectionArchitecture->getComputingUnitNb();
//
// T* sinogram_d;
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
// if (plan->sinogram_h->getUSinogramPixelNb() != uSinogramPixelNb)// sinogram_temp = new Sinogram3D_CPU<T>(uSinogramPixelNb, vSinogramPixelNb, projectionSinogramNb);
//
//
// //allocate device memory for sinogram result
// size_sinogram=uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb;
// size_tuile= uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMalloc((void**) &sinogram_d, sizeof(T)*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()*nstreams));//
//
// // //Copy constante
// // copy_constante_GPU_projection_ER(plan->constante_GPU_h,plan->sinogram_h->N_phi);
//
//
// //VOLUME MEMORY ALLOCATION
// //Create 3D array
// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(T), 0, 0, 0, cudaChannelFormatKindFloat);
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
//template <typename T>
//void RegularSamplingProjector_CPU<T>::projection_ER_semi_volume_kv_start_kv_stop(unsigned int semi_plan_z,unsigned int kv_start,unsigned int kv_stop,int size_sinogram,int size_tuile,TGPUplan_proj<Volume_CPU, Sinogram3D_CPU, T> *plan,T* sinogram_d,Sinogram3D_CPU<T> *sinogram_temp,cudaMemcpy3DParms *copyParams,cudaStream_t *streams,int nstreams)
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
// // plan->volume_h->saveVolume("/espace/boulay/volumeCPU.s");
//
// copyParams->srcPtr = make_cudaPitchedPtr(plan->volume_h->getVolumeData()+semi_plan_z*(N_zn_start)*plan->volume_h->getXVolumePixelNb()*plan->volume_h->getYVolumePixelNb(), copyParams->extent.width*sizeof(T), copyParams->extent.width, copyParams->extent.height);
// checkCudaErrors(cudaMemcpy3D(copyParams));
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
// projection_ERB_kernel_v0<<< dimGrid, dimBlock,0,streams[i]>>>(sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(),phi_start,N_un_start, N_vn_start, N_zn_start,uSinogramPixelNb,vSinogramPixelNb);
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
// T* coord_sino_h;
// T* coord_sino_d;
// coord_sino_h=plan->sinogram_h->getDataSinogram()+N_un_start+N_vn_start*plan->sinogram_h->getUSinogramPixelNb()+(unsigned long int)(phi_start+phi)*(unsigned long int)plan->sinogram_h->getUSinogramPixelNb()*(unsigned long int)plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_d=sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()+phi*uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMemcpyAsync(coord_sino_h,coord_sino_d, size_tuile*sizeof(T),cudaMemcpyDeviceToHost,streams[i]) );
// }
// }
// else
// {
// checkCudaErrors(cudaMemcpyAsync(sinogram_temp->getDataSinogram(),sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(), size_sinogram*sizeof(T),cudaMemcpyDeviceToHost,streams[i]));
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
// projection_ERB_kernel_v0<<< dimGrid, dimBlock,0,streams[i]>>>(sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(),phi_start,N_un_start, N_vn_start, N_zn_start,uSinogramPixelNb,vSinogramPixelNb);
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
// T *coord_sino_h;
// T *coord_sino_d;
// coord_sino_h=plan->sinogram_h->getDataSinogram()+N_un_start+N_vn_start*plan->sinogram_h->getUSinogramPixelNb()+(unsigned long int)(phi_start+phi)*(unsigned long int)plan->sinogram_h->getUSinogramPixelNb()*(unsigned long int)plan->sinogram_h->getVSinogramPixelNb();
// coord_sino_d=sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb()+phi*uSinogramPixelNb*vSinogramPixelNb;
// checkCudaErrors(cudaMemcpyAsync(coord_sino_h,coord_sino_d, size_tuile*sizeof(T),cudaMemcpyDeviceToHost,streams[i]) );
// }
// }
// else
// {
// checkCudaErrors(cudaMemcpyAsync(sinogram_temp->getDataSinogram(), sinogram_d+i*size_tuile*plan->cudaprojectionArchitecture->getProjectionThreadNb(), size_sinogram*sizeof(T),cudaMemcpyDeviceToHost,streams[i]) );
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

template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_CPU<T>::EnableP2P(){}

template <typename T>
void RegularSamplingProjector_compute_CUDA_mem_CPU<T>::DisableP2P(){}


template <typename T>
CUDAProjectionArchitecture* RegularSamplingProjector_compute_CUDA_mem_CPU<T>::getCUDAProjectionArchitecture() const
{
	return this->cudaprojectionArchitecture;
}

template <typename T>
__host__ void RegularSamplingProjector_compute_CUDA_mem_CPU<T>::copyConstantGPU()
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
	float GammaIOcylinderC = this->getGammaIOcylinderC();

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

	cudaMemcpyToSymbol(alphaIOcylinderC_GPU,this->getAlphaIOcylinderC(),projectionNb*sizeof(float));
	cudaMemcpyToSymbol(betaIOcylinderC_GPU,this->getBetaIOcylinderC(),projectionNb*sizeof(float));
	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&GammaIOcylinderC,sizeof(float));

}

template<typename T>
void RegularSamplingProjector_compute_CUDA_mem_CPU<T>::setCUDAProjectionArchitecture(CUDAProjectionArchitecture*  cudaprojectionArchitecture)
{
	this->cudaprojectionArchitecture =  cudaprojectionArchitecture;
}

#include "Projector_instances_CPU.cu"