/*
 * Convolution3D_CPU_GPU.cu
 *
 *      Author: gac
 */

#include "Convolution3D_CPU.cuh"
#include "GPUConstant.cuh"

#include "Convolution3D_kernel.cuh"

template <typename T>
Convolution3D_CPU<T>::Convolution3D_CPU(Image3D_CPU<T>* kernel): Convolution3D<T>(kernel){}

template <typename T>
Convolution3D_CPU<T>::Convolution3D_CPU(T* horizontalKernel, T* verticalKernel, T* depthKernel): Convolution3D<T>(horizontalKernel, verticalKernel, depthKernel){}

template <typename T>
Convolution3D_CPU<T>::~Convolution3D_CPU(){}

template <typename T>
CUT_THREADPROC Convolution3D_CPU<T>::solverThread(TGPUplan_conv3D<Volume_CPU, T> *plan)
{
	unsigned int kz,N_zn_par_carte,N_kz_par_carte,N_z_par_kernel;
	unsigned long int nb_elemt_bloc_kernel,size_bloc_kernel;

	int gpuNb = plan->gpuNb;

	unsigned long int xVolumePixelNb = plan->volume_in_h->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = plan->volume_in_h->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = plan->volume_in_h->getZVolumePixelNb();

	unsigned long int xyVolumePixelNb = xVolumePixelNb*yVolumePixelNb;

	T* volume_in_data = plan->volume_in_h->getVolumeData();
	T* volume_out_data = plan->volume_out_h->getVolumeData();

	cudaStream_t *streams;
	int nstreams;
	nstreams=plan->nstreams;
	//printf("Streams=%d %d\n",nstreams,plan->device);

	//Set device
	checkCudaErrors(cudaSetDevice(plan->device));

	N_zn_par_carte=zVolumePixelNb/gpuNb;
	N_z_par_kernel=BLOCK_SIZE_P_Z*NUMBER_COMPUTED_BLOCK;//64 z traité par kernel

	N_kz_par_carte=(unsigned int)(N_zn_par_carte/N_z_par_kernel);//nb de kernel par device

	nb_elemt_bloc_kernel= xyVolumePixelNb*N_z_par_kernel;
	size_bloc_kernel = nb_elemt_bloc_kernel*sizeof(T);

	cudaMemcpyToSymbol(c_volume_z, &(N_z_par_kernel),sizeof(T))	;

	streams = (cudaStream_t*)malloc((nstreams+1)*sizeof(cudaStream_t));

	for(int i=0; i<nstreams+1 ; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])) ;

	while (N_kz_par_carte%nstreams!=0){
		nstreams/=2;
	}

	//Traitement par slice en (x,y)
	dim3 ThreadsParBlock_h(BLOCK_SIZE_H_X ,BLOCK_SIZE_H_Y);
	dim3 BlocksParGrille_h(xVolumePixelNb/(ThreadsParBlock_h.x*NUMBER_COMPUTED_BLOCK),yVolumePixelNb*N_z_par_kernel/ThreadsParBlock_h.y);

	//Traitement par slice en (y,x)
	dim3 ThreadsParBlock_v(BLOCK_SIZE_V_X ,BLOCK_SIZE_V_Y );
	dim3 BlocksParGrille_v (xVolumePixelNb/ThreadsParBlock_v.x,(yVolumePixelNb*N_z_par_kernel)/(ThreadsParBlock_v.y*NUMBER_COMPUTED_BLOCK));

	//Traitement par slice en (z,x)
	dim3 ThreadsParBlock_p(BLOCK_SIZE_P_X ,BLOCK_SIZE_P_Z );
	dim3 BlocksParGrille_p(xVolumePixelNb/ThreadsParBlock_p.x,(N_z_par_kernel*yVolumePixelNb)/(ThreadsParBlock_p.y*NUMBER_COMPUTED_BLOCK));

	/*printf("%d BLOCK_SIZE_H_X=%d BLOCK_SIZE_H_Y=%d\n",plan->device,BLOCK_SIZE_H_X, BLOCK_SIZE_H_Y);
	printf("%d H BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_h.x,ThreadsParBlock_h.y,ThreadsParBlock_h.z,BlocksParGrille_h.x,BlocksParGrille_h.y,BlocksParGrille_h.z);
	printf("%d V BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_v.x,ThreadsParBlock_v.y,ThreadsParBlock_v.z,BlocksParGrille_v.x,BlocksParGrille_v.y,BlocksParGrille_v.z);
	printf("%d P BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_p.x,ThreadsParBlock_p.y,ThreadsParBlock_p.z,BlocksParGrille_p.x,BlocksParGrille_p.y,BlocksParGrille_p.z);
	 */
	T* d_BUFFER1;
	T* d_BUFFER2;

	checkCudaErrors(cudaMalloc((void**)&d_BUFFER1,(size_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb*sizeof(T))*nstreams));
	checkCudaErrors(cudaMalloc((void**)&d_BUFFER2,size_bloc_kernel*nstreams));
	checkCudaErrors(cudaMemset((void*)d_BUFFER2,0,size_bloc_kernel*nstreams));

	size_t *offset_volume_in;
	size_t *offset_volume_out;

	size_t *offset_buffer1;
	size_t *offset_buffer2;
	size_t *taille_transfert;
	char *up;
	char *down;

	offset_volume_in=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_volume_out=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_buffer1=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_buffer2=(size_t *)malloc(nstreams*sizeof(size_t));
	taille_transfert=(size_t *)malloc(nstreams*sizeof(size_t));
	up=(char *)malloc(nstreams*sizeof(char));
	down=(char *)malloc(nstreams*sizeof(char));


	for(kz=0;kz<N_kz_par_carte/nstreams;kz++){
		cudaEvent_t event;
		cudaEventCreate (&event);
		int i=0;
		//printf("%d kz=%d streams=%d soit %d\n",plan->device,kz,i,kz*nstreams+i);

		offset_volume_in[i]=(plan->device*N_zn_par_carte+(kz*nstreams+i)*N_z_par_kernel)*xyVolumePixelNb;

		offset_volume_out[i]=offset_volume_in[i];
		if (offset_volume_in[i]==0){//plancher du volume

			down[i]=0;
			if ((gpuNb==1) && (N_kz_par_carte==1))
				up[i]=0;
			else
				up[i]=1;
		}
		else{
			down[i]=1;

			if (offset_volume_in[i]==xyVolumePixelNb*zVolumePixelNb-xyVolumePixelNb*N_z_par_kernel)//plafond du volume
			{
				//printf("plafond \n");
				up[i]=0;
			}
			else {
				up[i]=1;
			}
		}
		taille_transfert[i]=size_bloc_kernel+(up[i]+down[i])*xyVolumePixelNb*sizeof(T);
		offset_buffer1[i]=down[i]*(BLOCK_SIZE_P_Z-plan->zKernelRadius)*(xyVolumePixelNb)+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb);
		offset_buffer2[i]=i*nb_elemt_bloc_kernel;
		offset_volume_in[i]-=down[i]*plan->zKernelRadius*(xyVolumePixelNb);

		checkCudaErrors(cudaMemcpyAsync(d_BUFFER1+offset_buffer1[i],volume_in_data+offset_volume_in[i],taille_transfert[i], cudaMemcpyHostToDevice,streams[i]));

		if (kz==0)
			cudaEventRecord (event, streams[i]);

		ConvKernel_h_shared  <<<BlocksParGrille_h, ThreadsParBlock_h,0,streams[i]>>> (d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
		ConvKernel_v_shared_Acc  <<<BlocksParGrille_v, ThreadsParBlock_v,0,streams[i]>>> (d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
		ConvKernel_p_shared_Acc  <<<BlocksParGrille_p, ThreadsParBlock_p,0,streams[i]>>> (d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i],up[i]);

		checkCudaErrors(cudaMemcpyAsync(volume_out_data+offset_volume_out[i], d_BUFFER2+offset_buffer2[i], size_bloc_kernel, cudaMemcpyDeviceToHost,streams[i]));

		for( i = 1 ; i < nstreams ; i++)
		{
			//printf("%d kz=%d streams=%d soit %d\n",plan->device,kz,i,kz*nstreams+i);
			offset_volume_in[i]=(plan->device*N_zn_par_carte+(kz*nstreams+i)*N_z_par_kernel)*xyVolumePixelNb;

			offset_volume_out[i]=offset_volume_in[i];
			if (offset_volume_in[i]==0){//plancher du volume

				down[i]=0;
				if ((gpuNb==1) && (N_kz_par_carte==1))
					up[i]=0;
				else
					up[i]=1;
			}
			else{
				down[i]=1;

				if (offset_volume_in[i]==xyVolumePixelNb*zVolumePixelNb-xyVolumePixelNb*N_z_par_kernel)//plafond du volume
				{
					//printf("plafond \n");
					up[i]=0;
				}
				else {
					up[i]=1;
				}
			}
			taille_transfert[i]=size_bloc_kernel+(up[i]+down[i])*xyVolumePixelNb*sizeof(T);
			offset_buffer1[i]=down[i]*(BLOCK_SIZE_P_Z-plan->zKernelRadius)*(xyVolumePixelNb)+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb);
			offset_buffer2[i]=i*nb_elemt_bloc_kernel;
			offset_volume_in[i]-=down[i]*plan->zKernelRadius*(xyVolumePixelNb);

			if (kz==0)
				cudaStreamWaitEvent ( streams[i], event,0 );

			checkCudaErrors(cudaMemcpyAsync(d_BUFFER1+offset_buffer1[i],volume_in_data+offset_volume_in[i],taille_transfert[i], cudaMemcpyHostToDevice,streams[i]));

			if (kz==0)
				cudaEventRecord (event, streams[i]);

			ConvKernel_h_shared<<<BlocksParGrille_h, ThreadsParBlock_h,0,streams[i]>>>(d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
			ConvKernel_v_shared_Acc<<<BlocksParGrille_v, ThreadsParBlock_v,0,streams[i]>>>(d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
			ConvKernel_p_shared_Acc<<<BlocksParGrille_p, ThreadsParBlock_p,0,streams[i]>>>(d_BUFFER2+i*nb_elemt_bloc_kernel,d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i],up[i]);

			checkCudaErrors(cudaMemcpyAsync(volume_out_data+offset_volume_out[i], d_BUFFER2+offset_buffer2[i],size_bloc_kernel,cudaMemcpyDeviceToHost,streams[i]));

		}//loop i streams
	}//loop kz

	checkCudaErrors(cudaFree(d_BUFFER1));
	checkCudaErrors(cudaFree(d_BUFFER2));

	for(int i = 0 ; i < nstreams ; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
}

/* Separable 3D Convolution on GPU */
template <typename T>
void Convolution3D_CPU<T>::doSeparableConvolution3D(Volume_CPU<T>* sourceImage, Volume_CPU<T>* convolutedImage)
{
	int nstreams=1;
	int gpuNb;
	if (sourceImage->getXVolumePixelNb()>=1024)
		cudaGetDeviceCount(&gpuNb);
	else
		gpuNb=1;
	//	this->copyConstantGPU(sourceImage);

	TGPUplan_conv3D<Volume_CPU,T> plan[MAX_GPU_COUNT];
	//CUTThread threadID[MAX_GPU_COUNT];
	std::thread **threadID;
	int device;

	struct cudaDeviceProp prop_device;
	cudaGetDeviceProperties(&prop_device,0);//propriétés du device 0

	float taille_allocation,taille_SDRAM,ratio_allocation_SDRAM;
	taille_SDRAM=(float)prop_device.totalGlobalMem;
	taille_allocation=2*sourceImage->getXVolumePixelNb()*sourceImage->getYVolumePixelNb()*sourceImage->getZVolumePixelNb()*sizeof(T);//il faut allouer 2 volumes
	ratio_allocation_SDRAM=taille_allocation/taille_SDRAM;
	//printf("allocation : %.2f Go  SDRAM : %.2f Go ratio :%.2f\n",taille_allocation/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_allocation_SDRAM);

	int nb_bloc=1;
	float taille_bloc_allocation,ratio_bloc_SDRAM;
	//printf("nb_blocs ");

	do{

		nb_bloc=2*nb_bloc;
		//printf("%d ",nb_bloc);
		taille_bloc_allocation=taille_allocation/nb_bloc;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	}
	while(ratio_bloc_SDRAM>=0.8);
	//printf("\n");

	threadID=(std::thread **)malloc(gpuNb*sizeof(std::thread *));

	for(device=0;device<gpuNb;device++)
	{
		plan[device].device=device;
		plan[device].volume_in_h=sourceImage;
		plan[device].volume_out_h=convolutedImage;
		plan[device].nstreams=nstreams;
		plan[device].zKernelRadius = this->getZKernelRadius();
		plan[device].gpuNb = gpuNb;

		//cout << "********** Start Constant Copy **********" << endl;
		//cout << "Convolution Constant Copy on device n° " << device << endl;
		cudaSetDevice(device);
		this->copyConstantGPU(sourceImage);
		//cout << "********** End Projection Constant Copy **********" << endl;
	}

	for(device = 0; device < gpuNb; device++)
	{
		//threadID[device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + device));
		threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + device));
	}

	//cutWaitForThreads(threadID,gpuNb);
	for (int i = 0; i < gpuNb; i++)
	{
		threadID[i]->join();
		delete threadID[i];
	}

	delete threadID;


}

Convolution3D_CPU_half::Convolution3D_CPU_half(Image3D_CPU<float>* kernel): Convolution3D_CPU<float>(kernel){}

Convolution3D_CPU_half::Convolution3D_CPU_half(float* horizontalKernel, float* verticalKernel, float* depthKernel): Convolution3D_CPU<float>(horizontalKernel, verticalKernel, depthKernel){}

Convolution3D_CPU_half::~Convolution3D_CPU_half(){}

CUT_THREADPROC Convolution3D_CPU_half::solverThread(TGPUplan_conv3D_half<Volume_CPU_half> *plan)
{

	unsigned int kz,N_zn_par_carte,N_kz_par_carte,N_z_par_kernel;
	unsigned long int nb_elemt_bloc_kernel,size_bloc_kernel;

	int gpuNb = plan->gpuNb;

	unsigned long int xVolumePixelNb = plan->volume_in_h->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = plan->volume_in_h->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = plan->volume_in_h->getZVolumePixelNb();

	unsigned long int xyVolumePixelNb = xVolumePixelNb*yVolumePixelNb;

	half* volume_in_data = plan->volume_in_h->getVolumeData();
	half* volume_out_data = plan->volume_out_h->getVolumeData();

	cudaStream_t *streams;
	int nstreams;
	nstreams=plan->nstreams;
	printf("Streams=%d %d\n",nstreams,plan->device);

	//Set device
	checkCudaErrors(cudaSetDevice(plan->device));

	N_zn_par_carte=zVolumePixelNb/gpuNb;
	N_z_par_kernel=BLOCK_SIZE_P_Z*NUMBER_COMPUTED_BLOCK;//64 z traité par kernel

	N_kz_par_carte=(unsigned int)(N_zn_par_carte/N_z_par_kernel);//nb de kernel par device

	nb_elemt_bloc_kernel= xyVolumePixelNb*N_z_par_kernel;
	size_bloc_kernel = nb_elemt_bloc_kernel*sizeof(half);

	cudaMemcpyToSymbol(c_volume_z, &(N_z_par_kernel),sizeof(half))	;

	streams = (cudaStream_t*)malloc((nstreams+1)*sizeof(cudaStream_t));

	for(int i=0; i<nstreams+1 ; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])) ;

	while (N_kz_par_carte%nstreams!=0){
		nstreams/=2;
	}

	//Traitement par slice en (x,y)
	dim3 ThreadsParBlock_h(BLOCK_SIZE_H_X ,BLOCK_SIZE_H_Y);
	dim3 BlocksParGrille_h(xVolumePixelNb/(ThreadsParBlock_h.x*NUMBER_COMPUTED_BLOCK),yVolumePixelNb*N_z_par_kernel/ThreadsParBlock_h.y);

	//Traitement par slice en (y,x)
	dim3 ThreadsParBlock_v(BLOCK_SIZE_V_X ,BLOCK_SIZE_V_Y );
	dim3 BlocksParGrille_v (xVolumePixelNb/ThreadsParBlock_v.x,(yVolumePixelNb*N_z_par_kernel)/(ThreadsParBlock_v.y*NUMBER_COMPUTED_BLOCK));

	//Traitement par slice en (z,x)
	dim3 ThreadsParBlock_p(BLOCK_SIZE_P_X ,BLOCK_SIZE_P_Z );
	dim3 BlocksParGrille_p(xVolumePixelNb/ThreadsParBlock_p.x,(N_z_par_kernel*yVolumePixelNb)/(ThreadsParBlock_p.y*NUMBER_COMPUTED_BLOCK));

	printf("%d BLOCK_SIZE_H_X=%d BLOCK_SIZE_H_Y=%d\n",plan->device,BLOCK_SIZE_H_X, BLOCK_SIZE_H_Y);
	printf("%d H BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_h.x,ThreadsParBlock_h.y,ThreadsParBlock_h.z,BlocksParGrille_h.x,BlocksParGrille_h.y,BlocksParGrille_h.z);
	printf("%d V BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_v.x,ThreadsParBlock_v.y,ThreadsParBlock_v.z,BlocksParGrille_v.x,BlocksParGrille_v.y,BlocksParGrille_v.z);
	printf("%d P BLOCK =%d,%d,%d GRID=%d,%d,%d\n",plan->device,ThreadsParBlock_p.x,ThreadsParBlock_p.y,ThreadsParBlock_p.z,BlocksParGrille_p.x,BlocksParGrille_p.y,BlocksParGrille_p.z);

	half* d_BUFFER1;
	half* d_BUFFER2;

	checkCudaErrors(cudaMalloc((void**)&d_BUFFER1,(size_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb*sizeof(half))*nstreams));
	checkCudaErrors(cudaMalloc((void**)&d_BUFFER2,size_bloc_kernel*nstreams));
	checkCudaErrors(cudaMemset((void*)d_BUFFER2,0,size_bloc_kernel*nstreams));

	size_t *offset_volume_in;
	size_t *offset_volume_out;

	size_t *offset_buffer1;
	size_t *offset_buffer2;
	size_t *taille_transfert;
	char *up;
	char *down;

	offset_volume_in=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_volume_out=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_buffer1=(size_t *)malloc(nstreams*sizeof(size_t));
	offset_buffer2=(size_t *)malloc(nstreams*sizeof(size_t));
	taille_transfert=(size_t *)malloc(nstreams*sizeof(size_t));
	up=(char *)malloc(nstreams*sizeof(char));
	down=(char *)malloc(nstreams*sizeof(char));

	for(kz=0;kz<N_kz_par_carte/nstreams;kz++){
		cudaEvent_t event;
		cudaEventCreate (&event);
		int i=0;
		printf("%d kz=%d streams=%d soit %d\n",plan->device,kz,i,kz*nstreams+i);

		offset_volume_in[i]=(plan->device*N_zn_par_carte+(kz*nstreams+i)*N_z_par_kernel)*xyVolumePixelNb;

		offset_volume_out[i]=offset_volume_in[i];
		if (offset_volume_in[i]==0){//plancher du volume

			down[i]=0;
			if ((gpuNb==1) && (N_kz_par_carte==1))
				up[i]=0;
			else
				up[i]=1;
		}
		else{
			down[i]=1;

			if (offset_volume_in[i]==xyVolumePixelNb*zVolumePixelNb-xyVolumePixelNb*N_z_par_kernel)//plafond du volume
			{
				printf("plafond \n");
				up[i]=0;
			}
			else {
				up[i]=1;
			}
		}
		taille_transfert[i]=size_bloc_kernel+(up[i]+down[i])*xyVolumePixelNb*sizeof(half);
		offset_buffer1[i]=down[i]*(BLOCK_SIZE_P_Z-plan->zKernelRadius)*(xyVolumePixelNb)+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb);
		offset_buffer2[i]=i*nb_elemt_bloc_kernel;
		offset_volume_in[i]-=down[i]*plan->zKernelRadius*(xyVolumePixelNb);

		checkCudaErrors(cudaMemcpyAsync(d_BUFFER1+offset_buffer1[i],volume_in_data+offset_volume_in[i],taille_transfert[i], cudaMemcpyHostToDevice,streams[i]));

		if (kz==0)
			cudaEventRecord (event, streams[i]);

		ConvKernel_h_shared_half  <<<BlocksParGrille_h, ThreadsParBlock_h,0,streams[i]>>> ((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
		ConvKernel_v_shared_Acc_half  <<<BlocksParGrille_v, ThreadsParBlock_v,0,streams[i]>>> ((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
		ConvKernel_p_shared_Acc_half  <<<BlocksParGrille_p, ThreadsParBlock_p,0,streams[i]>>> ((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i],up[i]);

		checkCudaErrors(cudaMemcpyAsync(volume_out_data+offset_volume_out[i], d_BUFFER2+offset_buffer2[i], size_bloc_kernel, cudaMemcpyDeviceToHost,streams[i]));

		for( i = 1 ; i < nstreams ; i++)
		{
			printf("%d kz=%d streams=%d soit %d\n",plan->device,kz,i,kz*nstreams+i);
			offset_volume_in[i]=(plan->device*N_zn_par_carte+(kz*nstreams+i)*N_z_par_kernel)*xyVolumePixelNb;

			offset_volume_out[i]=offset_volume_in[i];
			if (offset_volume_in[i]==0){//plancher du volume

				down[i]=0;
				if ((gpuNb==1) && (N_kz_par_carte==1))
					up[i]=0;
				else
					up[i]=1;
			}
			else{
				down[i]=1;

				if (offset_volume_in[i]==xyVolumePixelNb*zVolumePixelNb-xyVolumePixelNb*N_z_par_kernel)//plafond du volume
				{
					printf("plafond \n");
					up[i]=0;
				}
				else {
					up[i]=1;
				}
			}
			taille_transfert[i]=size_bloc_kernel+(up[i]+down[i])*xyVolumePixelNb*sizeof(half);
			offset_buffer1[i]=down[i]*(BLOCK_SIZE_P_Z-plan->zKernelRadius)*(xyVolumePixelNb)+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb);
			offset_buffer2[i]=i*nb_elemt_bloc_kernel;
			offset_volume_in[i]-=down[i]*plan->zKernelRadius*(xyVolumePixelNb);

			if (kz==0)
				cudaStreamWaitEvent ( streams[i], event,0 );

			checkCudaErrors(cudaMemcpyAsync(d_BUFFER1+offset_buffer1[i],volume_in_data+offset_volume_in[i],taille_transfert[i], cudaMemcpyHostToDevice,streams[i]));

			if (kz==0)
				cudaEventRecord (event, streams[i]);

			ConvKernel_h_shared_half<<<BlocksParGrille_h, ThreadsParBlock_h,0,streams[i]>>>((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
			ConvKernel_v_shared_Acc_half<<<BlocksParGrille_v, ThreadsParBlock_v,0,streams[i]>>>((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i]);
			ConvKernel_p_shared_Acc_half<<<BlocksParGrille_p, ThreadsParBlock_p,0,streams[i]>>>((unsigned short*)d_BUFFER2+i*nb_elemt_bloc_kernel,(unsigned short*)d_BUFFER1+i*(nb_elemt_bloc_kernel+2*BLOCK_SIZE_P_Z*xyVolumePixelNb),down[i],up[i]);

			checkCudaErrors(cudaMemcpyAsync(volume_out_data+offset_volume_out[i], d_BUFFER2+offset_buffer2[i],size_bloc_kernel,cudaMemcpyDeviceToHost,streams[i]));

		}//loop i streams
	}//loop kz

	checkCudaErrors(cudaFree(d_BUFFER1));
	checkCudaErrors(cudaFree(d_BUFFER2));

	for(int i = 0 ; i < nstreams ; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
}

/* Separable 3D Convolution on GPU */
void Convolution3D_CPU_half::doSeparableConvolution3D(Volume_CPU_half* sourceImage, Volume_CPU_half* convolutedImage)
{
	int gpuNb; cudaGetDeviceCount(&gpuNb);
	//	this->copyConstantGPU(sourceImage);
	int nstreams =4;
	TGPUplan_conv3D_half<Volume_CPU_half> plan[MAX_GPU_COUNT];
	//CUTThread threadID[MAX_GPU_COUNT];
	std::thread **threadID;
	int device;

	struct cudaDeviceProp prop_device;
	cudaGetDeviceProperties(&prop_device,0);//propriétés du device 0

	float taille_allocation,taille_SDRAM,ratio_allocation_SDRAM;
	taille_SDRAM=(float)prop_device.totalGlobalMem;
	taille_allocation=2*sourceImage->getXVolumePixelNb()*sourceImage->getYVolumePixelNb()*sourceImage->getZVolumePixelNb()*sizeof(half);//il faut allouer 2 volumes
	ratio_allocation_SDRAM=taille_allocation/taille_SDRAM;
	printf("allocation : %.2f Go  SDRAM : %.2f Go ratio :%.2f\n",taille_allocation/((1024.0*1024.0*1024.0)),taille_SDRAM/(1024.0*1024.0*1024.0),ratio_allocation_SDRAM);

	int nb_bloc=1;
	float taille_bloc_allocation,ratio_bloc_SDRAM;
	printf("nb_blocs ");

	do{

		nb_bloc=2*nb_bloc;
		printf("%d ",nb_bloc);
		taille_bloc_allocation=taille_allocation/nb_bloc;
		ratio_bloc_SDRAM=taille_bloc_allocation/taille_SDRAM;
	}
	while(ratio_bloc_SDRAM>=0.8);
	printf("\n");
	threadID=(std::thread **)malloc(gpuNb*sizeof(std::thread *));

	for(device=0;device<gpuNb;device++)
	{
		plan[device].device=device;
		plan[device].volume_in_h=sourceImage;
		plan[device].volume_out_h=convolutedImage;
		plan[device].nstreams=nstreams;
		plan[device].zKernelRadius = this->getZKernelRadius();
		plan[device].gpuNb = gpuNb;

		cout << "********** Start Constant Copy **********" << endl;
		cout << "Convolution Constant Copy on device n° " << device << endl;
		cudaSetDevice(device);
		this->copyConstantGPU((Volume<float> *)sourceImage);
		cout << "********** End Projection Constant Copy **********" << endl;
	}

	for(device = 0; device < gpuNb; device++)
	{
		//threadID[device] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + device));
		threadID[device] = new std::thread((CUT_THREADROUTINE)solverThread, (void *)(plan + device));
	}

	//cutWaitForThreads(threadID,gpuNb);
	for (int i = 0; i < gpuNb; i++)
	{
		threadID[i]->join();
		delete threadID[i];
	}

	delete threadID;
}


//#include "Convolution3D_instances.cu"
//#include "Convolution3D_instances_CPU.cu"


#include "Convolution3D_GPU.cuh"
//#include "GPUConstant.cuh"

//#include "Convolution3D_kernel.cuh"

template <typename T>
Convolution3D_GPU<T>::Convolution3D_GPU(Image3D_GPU<T>* kernel): Convolution3D<T>(kernel){} // Constructor for simple 3D convolution

template <typename T>
Convolution3D_GPU<T>::Convolution3D_GPU(T* horizontalKernel, T* verticalKernel, T* depthKernel): Convolution3D<T>(horizontalKernel, verticalKernel, depthKernel){}// Constructor for separable 3D convolution

template <typename T>
Convolution3D_GPU<T>::~Convolution3D_GPU(){}

/* Separable 3D Convolution on GPU */
template <typename T>
void Convolution3D_GPU<T>::doSeparableConvolution3D(Volume_GPU<T>* sourceImage, Volume_GPU<T>* convolutedImage)
{
	cout << "********** Start Separable 3D Volume Convolution all on GPU (float precision) **********" << endl;

	cudaSetDevice(gpuGetMaxGflopsDeviceId());
	this->copyConstantGPU(sourceImage);

	const unsigned long int xVolumeSize	= sourceImage->getXVolumePixelNb();
	const unsigned long int yVolumeSize = sourceImage->getYVolumePixelNb();
	const unsigned long int zVolumeSize = sourceImage->getZVolumePixelNb();

	T* sourceImageData = sourceImage->getVolumeImage()->getImageData();
	T* convolutedImageData = convolutedImage->getVolumeImage()->getImageData();

	//Slice processing (x,y)
	dim3 ThreadsParBlock_h(BLOCK_SIZE_H_X , BLOCK_SIZE_H_Y);
	dim3 BlocksParGrille_h(xVolumeSize/(ThreadsParBlock_h.x*NUMBER_COMPUTED_BLOCK), yVolumeSize*zVolumeSize/ThreadsParBlock_h.y);

	//Slice processing (y,x)
	dim3 ThreadsParBlock_v(BLOCK_SIZE_V_X ,BLOCK_SIZE_V_Y );
	dim3 BlocksParGrille_v (xVolumeSize/ThreadsParBlock_v.x, (yVolumeSize*zVolumeSize)/(ThreadsParBlock_v.y*NUMBER_COMPUTED_BLOCK));

	//Slice processing (z,x)
	dim3 ThreadsParBlock_p(BLOCK_SIZE_P_X ,BLOCK_SIZE_P_Z );
	dim3 BlocksParGrille_p(xVolumeSize/ThreadsParBlock_p.x, (yVolumeSize*zVolumeSize)/(ThreadsParBlock_p.y*NUMBER_COMPUTED_BLOCK));

	ConvKernel_h_shared  <<<BlocksParGrille_h, ThreadsParBlock_h>>> (convolutedImageData,sourceImageData,0);
	// DO NOT REMOVE !!!!
	checkCudaErrors(cudaDeviceSynchronize());

	ConvKernel_v_shared_Acc  <<<BlocksParGrille_v, ThreadsParBlock_v>>> (convolutedImageData,sourceImageData,0);
	// DO NOT REMOVE !!!!
	checkCudaErrors(cudaDeviceSynchronize());

	ConvKernel_p_shared_Acc  <<<BlocksParGrille_p, ThreadsParBlock_p>>> (convolutedImageData,sourceImageData,0,0);
	// DO NOT REMOVE !!!!
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "********** End Separable 3D Volume Convolution on GPU **********" << endl;
}

Convolution3D_GPU_half::Convolution3D_GPU_half(Image3D_GPU<float>* kernel): Convolution3D_GPU<float>(kernel){}

Convolution3D_GPU_half::Convolution3D_GPU_half(float* horizontalKernel, float* verticalKernel, float* depthKernel): Convolution3D_GPU<float>(horizontalKernel, verticalKernel, depthKernel){}

Convolution3D_GPU_half::~Convolution3D_GPU_half(){}

/* Separable 3D Convolution on GPU */
void Convolution3D_GPU_half::doSeparableConvolution3D(Volume_GPU_half* sourceImage, Volume_GPU_half* convolutedImage)
{
	cout << "********** Start Separable 3D Volume Convolution all on GPU (half-float precision) **********" << endl;

	cudaSetDevice(gpuGetMaxGflopsDeviceId());
	this->copyConstantGPU((Volume<float> *)sourceImage);

	const unsigned long int xVolumeSize	= sourceImage->getXVolumePixelNb();
	const unsigned long int yVolumeSize = sourceImage->getYVolumePixelNb();
	const unsigned long int zVolumeSize = sourceImage->getZVolumePixelNb();

	half* sourceImageData = sourceImage->getVolumeData();
	half* convolutedImageData = convolutedImage->getVolumeData();

	//Slice processing (x,y)
	dim3 ThreadsParBlock_h(BLOCK_SIZE_H_X , BLOCK_SIZE_H_Y);
	dim3 BlocksParGrille_h(xVolumeSize/(ThreadsParBlock_h.x*NUMBER_COMPUTED_BLOCK), yVolumeSize*zVolumeSize/ThreadsParBlock_h.y);

	//Slice processing (y,x)
	dim3 ThreadsParBlock_v(BLOCK_SIZE_V_X ,BLOCK_SIZE_V_Y );
	dim3 BlocksParGrille_v (xVolumeSize/ThreadsParBlock_v.x, (yVolumeSize*zVolumeSize)/(ThreadsParBlock_v.y*NUMBER_COMPUTED_BLOCK));

	//Slice processing (z,x)
	dim3 ThreadsParBlock_p(BLOCK_SIZE_P_X ,BLOCK_SIZE_P_Z );
	dim3 BlocksParGrille_p(xVolumeSize/ThreadsParBlock_p.x, (yVolumeSize*zVolumeSize)/(ThreadsParBlock_p.y*NUMBER_COMPUTED_BLOCK));

	ConvKernel_h_shared_half  <<<BlocksParGrille_h, ThreadsParBlock_h>>> ((unsigned short*)convolutedImageData,(unsigned short*)sourceImageData,0);
	// DO NOT REMOVE !!!!
	cudaDeviceSynchronize();


	ConvKernel_v_shared_Acc_half  <<<BlocksParGrille_v, ThreadsParBlock_v>>> ((unsigned short*)convolutedImageData,(unsigned short*)sourceImageData,0);
	// DO NOT REMOVE !!!!
	checkCudaErrors(cudaDeviceSynchronize());

	ConvKernel_p_shared_Acc_half  <<<BlocksParGrille_p, ThreadsParBlock_p>>> ((unsigned short*)convolutedImageData,(unsigned short*)sourceImageData,0,0);
	// DO NOT REMOVE !!!!
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "********** End Separable 3D Volume Convolution on GPU **********" << endl;
}

#include "Convolution3D_instances.cu"
#include "Convolution3D_instances_CPU.cu"
#include "Convolution3D_instances_GPU.cu"


