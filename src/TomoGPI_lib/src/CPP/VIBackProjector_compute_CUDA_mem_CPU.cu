/*
 * VIBackProjector_compute_CUDA_mem_CPU.cu
 *
  *      Author: gac
 */

#include "BackProjector_CPU.cuh"
#include "BackProjector_kernel.cuh"

/* VIBackProjector_CPU definition */
template<typename T>
VIBackProjector_compute_CUDA_mem_CPU<T>::VIBackProjector_compute_CUDA_mem_CPU(Acquisition* acquisition, Detector* detector, CUDABProjectionArchitecture *cudabackprojectionArchitecture,Volume_CPU<T>* volume,char fdk) : BackProjector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector,volume,fdk){
	this->setCUDABProjectionArchitecture(cudabackprojectionArchitecture);
}

template<typename T>
VIBackProjector_compute_CUDA_mem_CPU<T>::~VIBackProjector_compute_CUDA_mem_CPU(){}

template<typename T>
void VIBackProjector_compute_CUDA_mem_CPU<T>::doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
{
	std::cout << "\tVI BackProjection on CPU" << std::endl;
	TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>* plan;
	//CUTThread* threadID;
	std::thread **threadID;
	int device;

	this->setVolume(estimatedVolume);

	unsigned int gpuNb = this->getCUDABProjectionArchitecture()->getComputingUnitNb();
	int nstreams=this->getCUDABProjectionArchitecture()->getBProjectionStreamsNb();
	//printf("CUDA-capable device count: %d\n", gpuNb);

	struct cudaDeviceProp prop_device;
	cudaGetDeviceProperties(&prop_device,0);//propriétés du device 0
	int nb_bloc_phi=1;
	int nb_bloc_z_par_device=1;

	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
	float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	float vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();

	unsigned long int uSinogramPixelNb = sinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = sinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramNb = sinogram->getProjectionSinogramNb();

	unsigned long long int xThreadNb = this->getCUDABProjectionArchitecture()->getXThreadNb();
	unsigned long long int yThreadNb = this->getCUDABProjectionArchitecture()->getYThreadNb();
	unsigned long long int zThreadNb = this->getCUDABProjectionArchitecture()->getZThreadNb();

	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();
	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	unsigned long long int xBlockNb = this->getCUDABProjectionArchitecture()->getXBlockNb();
	unsigned long long int yBlockNb = this->getCUDABProjectionArchitecture()->getYBlockNb();
	unsigned long long int zBlockNb = this->getCUDABProjectionArchitecture()->getZBlockNb();


	float taille_bloc_allocation,ratio_bloc_SDRAM;
	unsigned int N_phi_par_bloc;
	float taille_allocation,taille_allocation_sino,taille_allocation_vol,taille_SDRAM,ratio_allocation_SDRAM;
	taille_SDRAM=(float)prop_device.totalGlobalMem;
	taille_allocation_sino=sizeof(T)*uSinogramPixelNb*vSinogramPixelNb*projectionSinogramNb/gpuNb;//il faut allouer 1 sinogramme
	taille_allocation_vol=nstreams*sizeof(T)*(size_t)(xThreadNb*xBlockNb)*(size_t)(yThreadNb*yBlockNb)*16.0;
	taille_allocation=taille_allocation_sino+taille_allocation_vol;
	ratio_allocation_SDRAM=taille_allocation/taille_SDRAM;
	//printf("allocation : %.2f Go  SDRAM : %.2f Go ratio :%.2f\n",taille_allocation/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_allocation_SDRAM);

	N_phi_par_bloc=projectionSinogramNb;
	while (N_phi_par_bloc>2048) //limitations des texture 2D par layer => pas plus de 2048 layers
	{
		nb_bloc_phi*=2;
		N_phi_par_bloc/=2;
	}

	//printf("nb_blocs_phi %d\n",nb_bloc_phi);
	//printf("nb_blocs_z_par_device %d ",nb_bloc_z_par_device);
	taille_bloc_allocation=taille_allocation_sino/(nb_bloc_phi*nb_bloc_z_par_device)+taille_allocation_vol;
	ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	//printf("allocation : %.2f Go  (sino %.2f Go vol %.2f Go)  SDRAM : %.2f Go ratio :%.2f\n",taille_bloc_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_sino/(nb_bloc_phi*nb_bloc_z_par_device))/((1024.0*1024.0*1024.0)),taille_allocation_vol/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_bloc_SDRAM);

	while(taille_allocation_vol/taille_SDRAM>=0.5)
	{
		if (nstreams>1)
			nstreams/=2;

		taille_allocation_vol=nstreams*sizeof(T)*(size_t)(xThreadNb*xBlockNb)*(size_t)(yThreadNb*yBlockNb)*16.0;;
		taille_allocation=taille_allocation_sino+taille_allocation_vol;
		taille_bloc_allocation=taille_allocation_sino/(nb_bloc_phi*nb_bloc_z_par_device)+taille_allocation_vol;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	}

	while(ratio_bloc_SDRAM>=0.7)
	{
		nb_bloc_z_par_device*=2;
		printf("\t%d ",nb_bloc_z_par_device);
		taille_bloc_allocation=taille_allocation_sino/(nb_bloc_phi*nb_bloc_z_par_device)+taille_allocation_vol;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
		printf("\tallocation : %.2f Go (sino %.2f Go vol %.2f Go) SDRAM : %.2f Go ratio :%.2f\n",taille_bloc_allocation/((1024.0*1024.0*1024.0)),(taille_allocation_sino/(nb_bloc_phi*nb_bloc_z_par_device))/((1024.0*1024.0*1024.0)),taille_allocation_vol/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_bloc_SDRAM);
	}

	unsigned int N_zn_par_carte;
	unsigned int N_zn_par_solverthread;
	unsigned int N_ligne_par_solverthread;
	unsigned int N_zn_par_kernel;
	unsigned int *num_bloc;
	num_bloc=(unsigned int*)malloc(sizeof(unsigned int)*nb_bloc_z_par_device*gpuNb);
	unsigned int *num_device;
	num_device=(unsigned int*)malloc(sizeof(unsigned int)*gpuNb);

	N_zn_par_carte=zVolumePixelNb/(gpuNb);
	N_zn_par_solverthread=N_zn_par_carte/(nb_bloc_z_par_device);
	N_zn_par_kernel=16;
	N_ligne_par_solverthread=(int)(N_zn_par_solverthread/N_zn_par_kernel);

	while (N_ligne_par_solverthread%nstreams!=0)
	{
		nstreams/=2;
	}

	this->getCUDABProjectionArchitecture()->setBProjectionStreamsNb(nstreams);

	//printf("N_zn_par_carte %d N_zn_par_solverthread %d nb_bloc_z_par_device %d \n",N_zn_par_carte,N_zn_par_solverthread,nb_bloc_z_par_device);
	cudaEvent_t *start_thread;
	cudaEvent_t *stop_thread;

	start_thread=(cudaEvent_t *)malloc(gpuNb*nb_bloc_z_par_device*sizeof(cudaEvent_t));
	stop_thread=(cudaEvent_t *)malloc(gpuNb*nb_bloc_z_par_device*sizeof(cudaEvent_t));

	plan=(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *)malloc(gpuNb*nb_bloc_z_par_device*sizeof(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>));
	//threadID=(CUTThread *)malloc(gpuNb*nb_bloc_z_par_device*sizeof(CUTThread));
	threadID=(std::thread **)malloc(gpuNb*sizeof(std::thread *));

	/*
	cout<<"\n \t sizeofT :"<<sizeof(T)<<endl;
	cout<<"\n \t nstreams :"<<nstreams<<endl;
	cout<<"\n \t gpuNb :"<<gpuNb<<endl;
	cout<<"\n \t vDetectorPixelSize :"<<vDetectorPixelSize<<endl;
	cout<<"\n \t vDetectorPixelNb :"<<vDetectorPixelNb<<endl;
	cout<<"\n \t vDetectorCenterPixel :"<<vDetectorCenterPixel<<endl;
	cout<<"\n \t uSinogramPixelNb :"<<uSinogramPixelNb<<endl;
	cout<<"\n \t vSinogramPixelNb :"<<vSinogramPixelNb<<endl;
	cout<<"\n \t projectionSinogramNb :"<<projectionSinogramNb<<endl;
	cout<<"\n \t xThreadNb :"<<xThreadNb<<endl;
	cout<<"\n \t yThreadNb :"<<yThreadNb<<endl;
	cout<<"\n \t zThreadNb :"<<zThreadNb<<endl;
	cout<<"\n \t xVolumePixelSize :"<<xVolumePixelSize<<endl;
	cout<<"\n \t zVolumeCenterPixel :"<<zVolumeCenterPixel<<endl;
	cout<<"\n \t xVolumePixelNb :"<<xVolumePixelNb<<endl;
	cout<<"\n \t yVolumePixelNb :"<<yVolumePixelNb<<endl;
	cout<<"\n \t zVolumePixelNb :"<<zVolumePixelNb<<endl;
	cout<<"\n \t xBlockNb :"<<xBlockNb<<endl;
	cout<<"\n \t yBlockNb :"<<yBlockNb<<endl;
	cout<<"\n \t zBlockNb :"<<zBlockNb<<endl;
	cout<<"\n \t taille_SDRAM :"<<taille_SDRAM<<endl;
	cout<<"\n \t taille_allocation_sino :"<<taille_allocation_sino<<endl;
	cout<<"\n \t taille_allocation_vol :"<<taille_allocation_vol<<endl;
	cout<<"\n \t taille_allocation :"<<taille_allocation<<endl;
	cout<<"\n \t ratio_allocation_SDRAM :"<<ratio_allocation_SDRAM<<endl;
	cout<<"\n \t N_phi_par_bloc :"<<N_phi_par_bloc<<endl;
	cout<<"\n \t taille_bloc_allocation :"<<taille_bloc_allocation<<endl;
	cout<<"\n \t ratio_bloc_SDRAM :"<<ratio_bloc_SDRAM<<endl;
	cout<<"\n \t N_zn_par_carte :"<<N_zn_par_carte<<endl;
	cout<<"\n \t N_zn_par_solverthread :"<<N_zn_par_solverthread<<endl;
	cout<<"\n \t N_zn_par_kernel :"<<N_zn_par_kernel<<endl;
	cout<<"\n \t N_ligne_par_solverthread :"<<N_ligne_par_solverthread<<endl;
	 */


	for(device=0;device<gpuNb;device++)
	{
		num_device[device]=device;

		//cout << "********** Start Constant Copy **********" << endl;
		//cout << "BackProjection Constant Copy on device n° " << device << endl;
		cudaSetDevice(device);
		this->copyConstantGPU();
		//cout << "********** End BackProjection Constant Copy **********" << endl;

		for (int n=0;n<nb_bloc_z_par_device;n++){
			if (device%2==1)
				num_bloc[n+device*nb_bloc_z_par_device]=(nb_bloc_z_par_device-1)-n;
			else
				num_bloc[n+device*nb_bloc_z_par_device]=n;
			//printf("n %d device %d num_device %d n+device*nb_bloc_z_par_device %d num_bloc %d\n",n,device,num_device[device],n+device*nb_bloc_z_par_device,num_bloc[n+device*nb_bloc_z_par_device]);
			checkCudaErrors(cudaEventCreate(start_thread+n+device*nb_bloc_z_par_device));
			checkCudaErrors(cudaEventCreate(stop_thread+n+device*nb_bloc_z_par_device));
			plan[n+device*nb_bloc_z_par_device].device=device;
			plan[n+device*nb_bloc_z_par_device].fdk=this->getFdk();
			plan[n+device*nb_bloc_z_par_device].volume_h=this->getVolume();
			plan[n+device*nb_bloc_z_par_device].sinogram_h=sinogram;
			plan[n+device*nb_bloc_z_par_device].acquisition=this->getAcquisition();
			plan[n+device*nb_bloc_z_par_device].detector=this->getDetector();
			plan[n+device*nb_bloc_z_par_device].cudabackprojectionArchitecture=this->getCUDABProjectionArchitecture();
			plan[n+device*nb_bloc_z_par_device].N_zn_par_carte=N_zn_par_carte;
			plan[n+device*nb_bloc_z_par_device].N_zn_par_solverthread=N_zn_par_solverthread;
		}
	}

	if (nb_bloc_phi==1){
		for (int n=0;n<nb_bloc_z_par_device;n++){
			for(device=0;device<gpuNb;device++){
				int zn_start;
				float zn_prime_start,zn_prime_stop;
				int vn_start,vn_stop;
				float vn_start_f,vn_stop_f;
				float vn_prime_start,vn_prime_stop;

				zn_start=num_bloc[n+device*nb_bloc_z_par_device]*N_zn_par_solverthread+num_device[device]*N_zn_par_carte;
				zn_prime_start=zn_start-zVolumeCenterPixel;
				zn_prime_stop=zn_prime_start+N_zn_par_solverthread;
				if(zn_prime_start>=0)
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				if(zn_prime_stop>=0)
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				vn_stop_f=vn_prime_stop+vDetectorCenterPixel;
				vn_stop=ceil(vn_stop_f);

				if (vn_stop<0)
					vn_stop=0;
				if (vn_stop>=vDetectorPixelNb)
					vn_stop=vDetectorPixelNb-1;

				vn_start_f=vn_prime_start+vDetectorCenterPixel;
				vn_start=floor(vn_start_f);

				if (vn_start<0)
					vn_start=0;

				//printf("device %d bloc %d zn_start %d zn_prime_start %f zn_prime_stop %f vn_prime_start %f vn_prime_stop %f vn_start %d vn_stop %d N_vn_par_solverthread %d\n",device,n,zn_start,zn_prime_start,zn_prime_stop,vn_prime_start,vn_prime_stop,vn_start,vn_stop,vn_stop-vn_start);

				plan[n+device*nb_bloc_z_par_device].zn_start=zn_start;
				plan[n+device*nb_bloc_z_par_device].vn_start=vn_start;
				plan[n+device*nb_bloc_z_par_device].N_vn_par_solverthread=(vn_stop-vn_start)+1;
				plan[n+device*nb_bloc_z_par_device].phi_start=0;
				plan[n+device*nb_bloc_z_par_device].N_phi_reduit=projectionSinogramNb;

				//threadID[device+n*nb_bloc_z_par_device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));
				threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));

				//std::cout << "Joinable after construction:\n" << std::boolalpha;
				//std::cout << device << " : " << threadID[device]->joinable() << '\n';

				/*
				cout<<"\t n :"<<n<<endl;
				cout<<"\t device :"<<device<<endl;
				cout<<"\t zn_start :"<<zn_start<<endl;
				cout<<"\t zVolumeCenterPixel :"<<zVolumeCenterPixel<<endl;
				cout<<"\t zn_prime_start :"<<zn_prime_start<<endl;
				cout<<"\t zn_prime_stop :"<<zn_prime_stop<<endl;
				cout<<"\t vn_start :"<<vn_start<<endl;
				cout<<"\t vn_stop :"<<vn_stop<<endl;
				cout<<"\t vn_start_f :"<<vn_start_f<<endl;
				cout<<"\t vn_stop_f :"<<vn_stop_f<<endl;
				cout<<"\t vn_prime_start :"<<vn_prime_start<<endl;
				cout<<"\t vn_prime_stop :"<<vn_prime_stop<<endl;
				 */

			}
			//cutWaitForThreads(threadID+n*nb_bloc_z_par_device,gpuNb);
			for (int i = 0; i < gpuNb; i++)
			{
				if (threadID[i]->joinable()){
					//std::cout << i <<" joined:\n" << endl;
					threadID[i]->join();
				}
			}
		}
	}
	else
	{
		Volume_CPU<T>* volume_temp_h = new Volume_CPU<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),(CUDAArchitecture *)NULL);


		for (int n=0;n<nb_bloc_z_par_device;n++){
			for(device=0;device<gpuNb;device++){
				int zn_start;
				float zn_prime_start,zn_prime_stop;
				int vn_start,vn_stop;
				float vn_start_f,vn_stop_f;
				float vn_prime_start,vn_prime_stop;

				zn_start=num_bloc[n+device*nb_bloc_z_par_device]*N_zn_par_solverthread+num_device[device]*N_zn_par_carte;
				zn_prime_start=zn_start-zVolumeCenterPixel;
				zn_prime_stop=zn_prime_start+N_zn_par_solverthread;
				if(zn_prime_start>=0)
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				if(zn_prime_stop>=0)
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				vn_stop_f=vn_prime_stop+vDetectorCenterPixel;
				vn_stop=ceil(vn_stop_f);

				if (vn_stop<0)
					vn_stop=0;
				if (vn_stop>=vDetectorPixelNb)
					vn_stop=vDetectorPixelNb-1;

				vn_start_f=vn_prime_start+vDetectorCenterPixel;
				vn_start=floor(vn_start_f);

				if (vn_start<0)
					vn_start=0;

				//printf("device %d bloc %d zn_start %d zn_prime_start %f zn_prime_stop %f vn_prime_start %f vn_prime_stop %f vn_start %d vn_stop %d N_vn_par_solverthread %d\n",device,n,zn_start,zn_prime_start,zn_prime_stop,vn_prime_start,vn_prime_stop,vn_start,vn_stop,vn_stop-vn_start);

				plan[n+device*nb_bloc_z_par_device].zn_start=zn_start;
				plan[n+device*nb_bloc_z_par_device].vn_start=vn_start;
				plan[n+device*nb_bloc_z_par_device].N_vn_par_solverthread=(vn_stop-vn_start)+1;
				plan[n+device*nb_bloc_z_par_device].phi_start=0;
				plan[n+device*nb_bloc_z_par_device].N_phi_reduit=N_phi_par_bloc;

				//threadID[device+n*nb_bloc_z_par_device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));
				threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));
			}
			//cutWaitForThreads(threadID+n*nb_bloc_z_par_device,gpuNb);
			for (int i = 0; i < gpuNb; i++)
			{
				threadID[i]->join();
			}
		}


		for (int n=0;n<nb_bloc_z_par_device;n++){
			for(device=0;device<gpuNb;device++){
				int zn_start;
				float zn_prime_start,zn_prime_stop;
				int vn_start,vn_stop;
				float vn_start_f,vn_stop_f;
				float vn_prime_start,vn_prime_stop;

				zn_start=num_bloc[n+device*nb_bloc_z_par_device]*N_zn_par_solverthread+num_device[device]*N_zn_par_carte;
				zn_prime_start=zn_start-zVolumeCenterPixel;
				zn_prime_stop=zn_prime_start+N_zn_par_solverthread;
				if(zn_prime_start>=0)
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				if(zn_prime_stop>=0)
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod-(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;
				else
					vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod+(float)(xVolumePixelNb)*xVolumePixelSize/2.0))/vDetectorPixelSize;

				vn_stop_f=vn_prime_stop+vDetectorCenterPixel;
				vn_stop=ceil(vn_stop_f);

				if (vn_stop<0)
					vn_stop=0;
				if (vn_stop>=vDetectorPixelNb)
					vn_stop=vDetectorPixelNb-1;

				vn_start_f=vn_prime_start+vDetectorCenterPixel;
				vn_start=floor(vn_start_f);

				if (vn_start<0)
					vn_start=0;

				//printf("device %d bloc %d zn_start %d zn_prime_start %f zn_prime_stop %f vn_prime_start %f vn_prime_stop %f vn_start %d vn_stop %d N_vn_par_solverthread %d\n",device,n,zn_start,zn_prime_start,zn_prime_stop,vn_prime_start,vn_prime_stop,vn_start,vn_stop,vn_stop-vn_start);

				plan[n+device*nb_bloc_z_par_device].volume_h=volume_temp_h;
				plan[n+device*nb_bloc_z_par_device].zn_start=zn_start;
				plan[n+device*nb_bloc_z_par_device].vn_start=vn_start;
				plan[n+device*nb_bloc_z_par_device].N_vn_par_solverthread=(vn_stop-vn_start)+1;
				plan[n+device*nb_bloc_z_par_device].phi_start=N_phi_par_bloc;
				plan[n+device*nb_bloc_z_par_device].N_phi_reduit=N_phi_par_bloc;

				//threadID[device+n*nb_bloc_z_par_device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));
				threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + n+device*nb_bloc_z_par_device));
			}
			//cutWaitForThreads(threadID+n*nb_bloc_z_par_device,gpuNb);
			for (int i = 0; i < gpuNb; i++)
			{
				threadID[i]->join();
			}
		}

		printf("add on CPU bloc\n");
		this->getVolume()->addVolume(volume_temp_h);
		delete volume_temp_h;
	}


	for (int i = 0; i < gpuNb; i++)
	{
		delete threadID[i];
	}

	delete threadID;
	delete plan;
	delete start_thread;
	delete stop_thread;
	delete num_bloc;
	delete num_device;
}

template <typename T>
CUT_THREADPROC VIBackProjector_compute_CUDA_mem_CPU<T>::solverThread(TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T> *plan)
{
	int phi_start=0;
	size_t size_volume;
	unsigned long int kligne;
	int zn_start;
	float zn_prime_start,zn_prime_stop;
	float vn_stop_f,vn_start_f;
	unsigned int vn_start_old,vn_stop_old;
	int vn_start,vn_stop,N_ligne_par_carte;
	float N_zn_restant,vn_prime_start,vn_prime_stop;
	T* volume_d;

	cudaChannelFormatDesc channelDesc;
	cudaArray *sino_cu_3darray;
	cudaMemcpy3DParms myparms_sino_3Darray = {0};

	unsigned int gpuNb = plan->cudabackprojectionArchitecture->getComputingUnitNb();
	cudaStream_t *streams;
	int nstreams=plan->cudabackprojectionArchitecture->getBProjectionStreamsNb();

	float fdd = plan->acquisition->getFocusDetectorDistance();
	float fod = plan->acquisition->getFocusObjectDistance();
	float vDetectorPixelSize = plan->detector->getVDetectorPixelSize();
	float vDetectorPixelNb = plan->detector->getVDetectorPixelNb();
	float vDetectorCenterPixel = plan->detector->getVDetectorCenterPixel();

	float xVolumePixelSize = plan->volume_h->getXVolumePixelSize();
	float zVolumeCenterPixel = plan->volume_h->getZVolumeCenterPixel();
	unsigned long int xVolumePixelNb = plan->volume_h->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = plan->volume_h->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = plan->volume_h->getZVolumePixelNb();

	unsigned short xThreadNb = plan->cudabackprojectionArchitecture->getXThreadNb();
	unsigned short yThreadNb = plan->cudabackprojectionArchitecture->getYThreadNb();
	unsigned short zThreadNb = plan->cudabackprojectionArchitecture->getZThreadNb();
	unsigned short xBlockNb = plan->cudabackprojectionArchitecture->getXBlockNb();
	unsigned short yBlockNb = plan->cudabackprojectionArchitecture->getYBlockNb();
	unsigned short zBlockNb = plan->cudabackprojectionArchitecture->getZBlockNb();

	unsigned long int sinoU_h = plan->sinogram_h->getUSinogramPixelNb();
	unsigned long int sinoV_h = plan->sinogram_h->getVSinogramPixelNb();
	unsigned long int sinoPhi_h = plan->sinogram_h->getProjectionSinogramNb();

	T* dataSinogram = plan->sinogram_h->getDataSinogram();
	T* dataVolume = plan->volume_h->getVolumeData();

	//Set device
	checkCudaErrors(cudaSetDevice(plan->device));

	cudaEvent_t start_solverthread,stop_solverthread;
	checkCudaErrors(cudaEventCreate(&start_solverthread));
	checkCudaErrors(cudaEventCreate(&stop_solverthread));
	checkCudaErrors(cudaEventRecord(start_solverthread, NULL));
	streams = (cudaStream_t*)malloc((nstreams+1)*sizeof(cudaStream_t));

	for(int i=0; i<nstreams+1 ; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])) ;

	N_ligne_par_carte=(int)(plan->N_zn_par_solverthread/16);

	cudaEvent_t *event;
	event=(cudaEvent_t *)malloc(nstreams*sizeof(cudaEvent_t));
	for(int i=0;i<nstreams;i++)
		checkCudaErrors(cudaEventCreate(event+i));

	N_zn_restant=((float)(plan->N_zn_par_solverthread)/16)-N_ligne_par_carte;

	if(N_zn_restant>0)
	{
		N_ligne_par_carte+=1;
	}
	//printf("GPU%d N_zn_par_solverthread: %d N_ligne_par_carte:%d N_zn_restant; %f\n",plan->device,plan->N_zn_par_solverthread, N_ligne_par_carte,N_zn_restant);

	//Decoupage en thread
	dim3 dimBlock(xThreadNb,yThreadNb,zThreadNb);
	dim3 dimGrid(xBlockNb,yBlockNb,zBlockNb);
	size_volume=(size_t)(xThreadNb*xBlockNb)*(size_t)(yThreadNb*yBlockNb)*(size_t)16;
	checkCudaErrors(cudaMalloc((void**) &(volume_d), sizeof(T)*size_volume*(size_t)nstreams));

	//Mise des sinogram en texture 2D layered
	channelDesc = cudaCreateChannelDesc(sizeof(T)*8, 0, 0, 0, cudaChannelFormatKindFloat);

	checkCudaErrors(cudaMalloc3DArray(&sino_cu_3darray, &channelDesc, make_cudaExtent((size_t)sinoU_h,(size_t)plan->N_vn_par_solverthread,(size_t)plan->N_phi_reduit), cudaArrayLayered));

	//cout<<"  device "<<plan->device<<"  "<<sinoU_h<<"   "<<plan->N_vn_par_solverthread<<"    "<<plan->N_phi_reduit<<endl;

	myparms_sino_3Darray.kind = cudaMemcpyHostToDevice;
	myparms_sino_3Darray.dstArray = sino_cu_3darray;


	sinogram_tex0.addressMode[0] = cudaAddressModeBorder;
	sinogram_tex0.addressMode[1] = cudaAddressModeBorder;
	sinogram_tex0.filterMode = cudaFilterModeLinear;
	sinogram_tex0.normalized = false; // access with normalized texture coordinates

	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(sinogram_tex0, sino_cu_3darray, channelDesc));

	phi_start=plan->phi_start;
	kligne=0;

	checkCudaErrors(cudaMemset((void*) volume_d, 0,sizeof(T)*size_volume*nstreams));

	for(kligne=0;kligne<((N_ligne_par_carte/nstreams));kligne++)
	{
		vn_stop_old=vn_stop;

		int i = 0;

		zn_start=plan->zn_start+(i+kligne*nstreams)*16;
		zn_prime_start=zn_start-zVolumeCenterPixel;
		zn_prime_stop=zn_prime_start+16;
		if(zn_prime_stop>=0){
			vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;
		}
		else
		{
			vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;
		}

		vn_stop_f=vn_prime_stop+vDetectorCenterPixel;
		vn_stop=ceil(vn_stop_f);

		if (vn_stop<0)
			vn_stop=0;
		if (vn_stop>=vDetectorPixelNb)
			vn_stop=vDetectorPixelNb-1;

		if(kligne!=0){
			vn_start=vn_stop_old+1;
		}
		else{
			if(zn_prime_start>=0){
				vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;
			}
			else
			{
				vn_prime_start=((float)fdd*(float)zn_prime_start*xVolumePixelSize/((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;
			}
			vn_start_f=vn_prime_start+vDetectorCenterPixel;
			vn_start=floor(vn_start_f);

			if (vn_start<0)
				vn_start=0;
			if ((vn_start-plan->vn_start)<0)
				vn_start=plan->vn_start;


		}


		//printf("%d zn_start %d zn_prime_start %f zn_prime_stop %f vn_prime_start %f vn_prime_stop %f vn_start %d vn_stop %d(vn_stop-vn_start)+1 %d\n",i,zn_start,zn_prime_start,zn_prime_stop,vn_prime_start,vn_prime_stop,vn_start,vn_stop,(vn_stop-vn_start)+1);

		if (kligne!=0)
			cudaStreamWaitEvent( streams[i], event[nstreams-1],0 );


		//cout << "---Load device/ i  /phi/minphi/maxphi " << plan->device << " " << i  <<" " << " " << plan->phi_start <<" "<< plan->N_phi_reduit << endl;
		//cout << "--- v_start / Plan_v/v_stop  "<< vn_start << " " << plan->vn_start<< "  " << vn_stop << endl;


		for (unsigned int phi=plan->phi_start;phi<plan->phi_start+plan->N_phi_reduit;phi++)
		{
			myparms_sino_3Darray.srcPos = make_cudaPos(0,0,0);//make_cudaPos(0,plan->vn_start,phi);
			myparms_sino_3Darray.dstPos = make_cudaPos(0,vn_start-plan->vn_start,phi-plan->phi_start);
			myparms_sino_3Darray.srcPtr = make_cudaPitchedPtr(dataSinogram+phi*sinoU_h*sinoV_h+vn_start*sinoU_h, sinoU_h*sizeof(T), sinoU_h, sinoV_h);
			myparms_sino_3Darray.extent = make_cudaExtent(sinoU_h,(vn_stop-vn_start)+1,1);

			if (((vn_stop-vn_start)+1)>0)
			{
				checkCudaErrors(cudaMemcpy3DAsync(&myparms_sino_3Darray,streams[i]));
				//cout<<"\n\n copy"<<vn_stop<<"  "<<vn_start<<endl;
			}
		}

		cudaEventRecord (event[i], streams[i]);
		cudaError_t error;
		cudaEvent_t start,stop;
		error = cudaEventCreate(&start);
		error = cudaEventCreate(&stop);
	
		// Record the start event
		error = cudaEventRecord(start, NULL);
		error = cudaEventSynchronize(start);

		if (plan->fdk)
			FDK3D_VIB_kernel_v0_16reg<<< dimGrid, dimBlock,0,streams[i]>>>(volume_d+size_volume*i,phi_start,zn_start,plan->N_phi_reduit,plan->vn_start);
		else
			backprojection_VIB_kernel_v2_16reg<<< dimGrid, dimBlock,0,streams[i]>>>(volume_d+size_volume*i,phi_start,zn_start,plan->N_phi_reduit,plan->vn_start);


			error = cudaEventRecord(stop, NULL);
			// Wait for the stop event to complete
			error = cudaEventSynchronize(stop);
			float msecTotal = 0.0f;
			error = cudaEventElapsedTime(&msecTotal, start, stop);
		
		
			printf("Backproj execution time %f\n",msecTotal);

		checkCudaErrors(cudaMemcpyAsync(dataVolume+zn_start*xVolumePixelNb*yVolumePixelNb, volume_d+size_volume*i, size_volume*sizeof(T),cudaMemcpyDeviceToHost,streams[i])) ;

		for(i=1; i < nstreams; i++)
		{
			vn_stop_old=vn_stop;
			zn_start=plan->zn_start+(i+kligne*nstreams)*16;

			zn_prime_start=zn_start-zVolumeCenterPixel;
			zn_prime_stop=zn_prime_start+16;

			if(zn_prime_stop>=0)
				vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod-(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;
			else
				vn_prime_stop=((float)fdd*(float)zn_prime_stop*xVolumePixelSize/((float)fod+(float)xVolumePixelNb*xVolumePixelSize/2.0))/vDetectorPixelSize;

			vn_stop_f=vn_prime_stop+vDetectorCenterPixel;
			vn_stop=ceil(vn_stop_f);

			if (vn_stop<0)
				vn_stop=0;
			if (vn_stop>=vDetectorPixelNb)
				vn_stop=vDetectorPixelNb-1;

			vn_start=vn_stop_old+1;

			//printf("%i zn_start %d zn_prime_start %f zn_prime_stop %f vn_prime_start %f vn_prime_stop %f vn_start %d vn_stop %d (vn_stop-vn_start)+1 %d\n",i,zn_start,zn_prime_start,zn_prime_stop,vn_prime_start,vn_prime_stop,vn_start,vn_stop,(vn_stop-vn_start)+1);

			if (((vn_stop-vn_start)+1)>0)
				cudaStreamWaitEvent( streams[i], event[i-1],0 );

			//cout << "+++ Load device/ i  /phi/minphi/maxphi " << plan->device << " " << i  << " " << plan->phi_start <<" "<< plan->N_phi_reduit << endl;
			//cout << "+++ v_start / Plan_v/v_stop  "<< vn_start << " " << plan->vn_start<< "  " << vn_stop << endl;

			for (unsigned int phi=plan->phi_start;phi<plan->phi_start+plan->N_phi_reduit;phi++){

				myparms_sino_3Darray.srcPos = make_cudaPos(0,0,0);
				myparms_sino_3Darray.dstPos = make_cudaPos(0,vn_start-plan->vn_start,phi-plan->phi_start);
				myparms_sino_3Darray.srcPtr = make_cudaPitchedPtr(dataSinogram+phi*sinoU_h*sinoV_h+vn_start*sinoU_h, sinoU_h*sizeof(T), sinoU_h,sinoV_h);
				myparms_sino_3Darray.extent = make_cudaExtent(sinoU_h,(vn_stop-vn_start)+1,1);

				if (((vn_stop-vn_start)+1)>0)
					checkCudaErrors(cudaMemcpy3DAsync(&myparms_sino_3Darray,streams[i]));
			}

			cudaEventRecord (event[i], streams[i]);


			if (plan->fdk)
				FDK3D_VIB_kernel_v0_16reg<<< dimGrid, dimBlock,0,streams[i]>>>(volume_d+size_volume*i,phi_start,zn_start,plan->N_phi_reduit,plan->vn_start);
			else
				backprojection_VIB_kernel_v2_16reg<<< dimGrid, dimBlock,0,streams[i]>>>(volume_d+size_volume*i,phi_start,zn_start,plan->N_phi_reduit,plan->vn_start);

			checkCudaErrors(cudaMemcpyAsync(dataVolume+zn_start*xVolumePixelNb*yVolumePixelNb, volume_d+size_volume*i, size_volume*sizeof(T),cudaMemcpyDeviceToHost,streams[i])) ;
		}
	}

	checkCudaErrors(cudaFreeArray(sino_cu_3darray));
	cudaFree(volume_d);

	for(int i=0 ; i < nstreams ; i++)
	{
		cudaEventDestroy(event[i]);
	}
	free(event);

	for(int i = 0 ; i < nstreams ; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));

	free(streams);

	checkCudaErrors(cudaEventRecord(stop_solverthread, NULL));
	checkCudaErrors(cudaEventSynchronize(stop_solverthread));

	cudaEventDestroy(start_solverthread);
	cudaEventDestroy(stop_solverthread);
}


template <typename T>
void VIBackProjector_compute_CUDA_mem_CPU<T>::EnableP2P(){}

template <typename T>
void VIBackProjector_compute_CUDA_mem_CPU<T>::DisableP2P(){}


template<typename T>
CUDABProjectionArchitecture* VIBackProjector_compute_CUDA_mem_CPU<T>::getCUDABProjectionArchitecture() const
{
	return this->cudabackprojectionArchitecture;
}

template <typename T>
__host__ void VIBackProjector_compute_CUDA_mem_CPU<T>::copyConstantGPU()
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

	float gammaIOcylinderC= this->getGammaIOcylinderC();
	float alphaC= this->getAlphaC();
	float betaC= this->getBetaC();

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
	cudaMemcpyToSymbol(gammaIOcylinderC_GPU,&gammaIOcylinderC,sizeof(float));

	cudaMemcpyToSymbol(alphaC_GPU,&alphaC,sizeof(float));
	cudaMemcpyToSymbol(betaC_GPU,&betaC,sizeof(float));
}

template<typename T>
void VIBackProjector_compute_CUDA_mem_CPU<T>::setCUDABProjectionArchitecture(CUDABProjectionArchitecture*  cudabackprojectionArchitecture)
{
	this->cudabackprojectionArchitecture =  cudabackprojectionArchitecture;
}

//#include "BackProjector_instances.cu"
#include "BackProjector_instances_CPU.cu"
