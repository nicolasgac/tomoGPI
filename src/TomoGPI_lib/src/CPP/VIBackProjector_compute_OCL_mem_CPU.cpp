/*
 * VIBackProjector_compute_OCL_mem_CPU.cpp
 *
 *      Author: diakite
 */

#include "BackProjector_CPU.cuh"

#define SOURCE_FILE "TomoBayes/src/TomoBayes_lib/src/OCL/backprojection3D_kernel.cl"


//#include "opencl_compat.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif
//#include "AOCLUtils/aocl_utils.h"

//#include "backprojection3D_kernel.cl"

//OLD API
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define AOCL_ALIGNMENT 64  //Memory aligned 64 bytes to use DMA for transfert

#define CLEANUP()                                     \
    do {                                              \
        ret = clFlush(queue);             \
        ret = clFinish(queue);            \
        ret = clReleaseKernel(kernel);                \
        ret = clReleaseProgram(program);              \
        ret = clReleaseCommandQueue(queue);   \
        ret = clReleaseContext(context);              \
        free(source_str);                             \
    } while(0)
    
#define CHECK_RET(ret, str)         \
do {                            \
    if(ret != CL_SUCCESS) {     \
        puts(str);              \
        printf("%d\n", ret);    \
        CLEANUP();              \
    }                           \
} while(0)


typedef struct struct_sampling_opencl1 {
  //! Taille du voxel du volume en x
  float delta_xn; 

  float xn_0;
  //! (N_yn_FOV / 2) - 0.5
  float yn_0;
  //! (N_zn_FOV / 2) - 0.5
  float zn_0;   ///15

  int N_un;
  //!  Nombre de pixels du plan detecteur en v
  int N_vn;

  int N_xn_FOV;
} type_struct_sampling_opencl1;

//#define PHI_MAX 1000
typedef struct struct_constante_opencl1 {
  float alpha_wn[PHI_MAX];
  float beta_wn[PHI_MAX];
  float gamma_wn;
  float gamma_vn;
  float D;
  float delta_un;
  float gamma_D;//20
  float un_0;
  float vn_0;
}type_struct_constante_opencl1;


template<typename T>
VIBackProjector_compute_OCL_mem_CPU<T>::VIBackProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector, OCLBProjectionArchitecture *oclbackprojectionArchitecture, Volume_CPU<T>* volume,char fdk) : BackProjector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector,volume,fdk){
    this->setOCLBProjectionArchitecture(oclbackprojectionArchitecture);
}

template<typename T>
VIBackProjector_compute_OCL_mem_CPU<T>::~VIBackProjector_compute_OCL_mem_CPU(){}


template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
{
    float * host_volume = (float *)malloc(256*256*256*sizeof(float));
    float * host_volume1 = (float *)malloc(256*256*256*sizeof(float));
    
    switch (this->oclbackprojectionArchitecture->getArchitecture())
    {
    case ARCHITECTURE_FPGA:
        printf("Backprojection FPGA OCL\n");
        this->doBackProjection_FPGA(estimatedVolume, sinogram);
        break;
    case ARCHITECTURE_GPU:
        printf("Backprojection GPU OCL\n");
        this->doBackProjection_GPU(estimatedVolume, sinogram);
        break;
    case ARCHITECTURE_CPU:
        printf("Backprojection CPU OCL\n");
        // this->doBackProjection_GPU(estimatedVolume, sinogram);
        // memcpy(host_volume,estimatedVolume->getVolumeData(),estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float));
        this->doBackProjection_CPU(estimatedVolume, sinogram);
        memcpy(host_volume1,estimatedVolume->getVolumeData(),estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float));
        float RMSE;
        RMSE = 0.0;
        for(unsigned int p=0; p<256*256*256; p++){
            //if(host_volume1[p]!=0){
            RMSE += (host_volume1[p] - host_volume[p]) * (host_volume1[p] - host_volume[p]);
            //}
            //printf("%f = %f\t", host_volume1[p], host_volume[p]);
        }
        RMSE = sqrt(RMSE/(256*256*256));
        printf("Erreur quadratique moyenne: %.3f\n", RMSE);
        break;
    
    default:
        break;
    }
      
    
}


template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::doBackProjection_FPGA(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
{
	std::cout << "\tVI OpenCL BackProjection " << std::endl;
	TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>* plan;

	this->setVolume(estimatedVolume);
    
	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
    float uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
    float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();

	unsigned long int uSinogramPixelNb = sinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = sinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramNb = sinogram->getProjectionSinogramNb();

	unsigned long long int xThreadNb = this->getOCLBProjectionArchitecture()->getXThreadNb();
	unsigned long long int yThreadNb = this->getOCLBProjectionArchitecture()->getYThreadNb();
	unsigned long long int zThreadNb = this->getOCLBProjectionArchitecture()->getZThreadNb();

	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float yVolumePixelSize = this->getVolume()->getYVolumePixelSize();
	float zVolumePixelSize = this->getVolume()->getZVolumePixelSize();

	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();

	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	unsigned long long int xBlockNb = this->getOCLBProjectionArchitecture()->getXBlockNb();
	unsigned long long int yBlockNb = this->getOCLBProjectionArchitecture()->getYBlockNb();
	unsigned long long int zBlockNb = this->getOCLBProjectionArchitecture()->getZBlockNb();

	//CODE OPENCL PROVENANT DE TOMOX

    // Initialisation des objets OpenCL
    static cl_platform_id *platform_id = NULL;
    static cl_device_id device_id = NULL;
    static cl_context context = NULL;
    static cl_command_queue queue;
    static cl_kernel kernel;
    static cl_program program = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_event kernel_event;

    //Lecture du .aocx
    unsigned char *source_str;
    size_t source_size;
    FILE *fp;
    char fileName[] = "backprojection3D_kernel.aocx";
    fp = fopen(fileName, "rb");
    if (!fp) {
    	   fprintf(stderr, "Failed to load kernel.\n");
         exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (unsigned char*) malloc(source_size * sizeof(unsigned char));
    if (fread(source_str, 1, source_size, fp) == 0) {
        puts("Could not read source file");
        exit(-1);
    }
    printf("Taille du binaire : %lu bytes\n", source_size);
    fclose(fp);

  
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    ret = clGetDeviceIDs(platform_id[2], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

   // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_RET(ret, "Create Context");

    cl_event user_event = clCreateUserEvent(context, &ret);

    double elapsed = 0;
    cl_ulong time_start, time_end;

    // Obligation de recopie de certaines donnees : OpenCL ne gere pas la recopie de pointeurs de pointeurs	
	type_struct_constante_opencl1 * host_constant = (type_struct_constante_opencl1 * ) malloc(sizeof(type_struct_constante_opencl1));
	type_struct_sampling_opencl1* host_sampling = (type_struct_sampling_opencl1*) malloc (sizeof(type_struct_sampling_opencl1));

    
    void * host_volume = NULL; //(cl_float*)alignedMalloc(volume->N_xn*volume->N_yn*volume->N_zn*sizeof(float));
	posix_memalign ((void **)&host_volume, AOCL_ALIGNMENT, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float));
    
	//float * host_volume = (float *)estimatedVolume->getVolumeData();
	//printf("Taille host_volume : %d * sizeof(float)\n",estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb());

    
    void * host_sinogram = NULL; //(cl_float*)alignedMalloc(sinogram->N_phi*sinogram->N_un*sinogram->N_vn*sizeof(float));
	posix_memalign ((void **)&host_sinogram, AOCL_ALIGNMENT, sinogram->getDataSinogramSize()*sizeof(float));

    memcpy(host_volume,estimatedVolume->getVolumeData(),estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float));
   	memcpy(host_sinogram,sinogram->getDataSinogram(),sinogram->getDataSinogramSize()*sizeof(float));
    
    cl_float2 * alpha_beta = NULL; // alpha and beta in a vector type float2
    posix_memalign ((void **)&alpha_beta, AOCL_ALIGNMENT, sizeof(cl_float2)*projectionSinogramNb);


    //float * host_sinogram = (float *)sinogram->getDataSinogram();
	//printf("Taille host_sinogram : %d * sizeof(float)\n",sinogram->getDataSinogramSize());
 
	cl_ulong3 host_vol_dims;
	cl_ulong3 host_sin_dims;

	// Division effectuee du cote hote
	float host_cst = xVolumePixelSize * (fdd / uDetectorPixelSize);
	printf("Valeur host_const : %f\n", host_cst);
    

	host_vol_dims.x=xVolumePixelNb;
	host_vol_dims.y=yVolumePixelNb;
	host_vol_dims.z=zVolumePixelNb;

	host_sin_dims.x=uSinogramPixelNb;
	host_sin_dims.y=vSinogramPixelNb;
	host_sin_dims.z=projectionSinogramNb;

	host_sampling->delta_xn=xVolumePixelSize;
	host_sampling->N_un=uDetectorPixelNb;
	host_sampling->N_vn=vDetectorPixelNb;

	host_sampling->xn_0=xVolumeCenterPixel;
	host_sampling->yn_0=yVolumeCenterPixel;
	host_sampling->zn_0=zVolumeCenterPixel;

	host_constant->gamma_wn=fod;
	host_constant->gamma_vn=1.0/this->getGammaIOcylinderC(); //(sampling->D*sampling->delta_zn)/(sampling->delta_vn);
	host_constant->delta_un=uDetectorPixelSize;
	host_constant->un_0=uDetectorCenterPixel;
	host_constant->vn_0=vDetectorCenterPixel;
	host_constant->gamma_D=(vDetectorPixelSize*fod)/(fdd*zVolumePixelSize);//(sampling->delta_vn*sampling->R)/(sampling->D*sampling->delta_zn);
	host_constant->D=fdd;

    printf("test_value: %f\n", host_constant->vn_0);

    
    for (int p=0; p<projectionSinogramNb;p++){
	    //host_constant->alpha_wn[p] = this->getAlphaIOcylinderC()[p];
	    //host_constant->beta_wn[p] = this->getBetaIOcylinderC()[p];
        alpha_beta[p].s0=this->getAlphaIOcylinderC()[p]; // vector first element is alpha
        alpha_beta[p].s1=this->getBetaIOcylinderC()[p];  // vector second element is beta
       // printf("%f\t", host_constant->alpha_wn[p]);
    }

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem device_volume = clCreateBuffer(context, CL_MEM_READ_WRITE, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), NULL,&ret);
	printf("device_volume created\n");
	//cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram,&ret);
    cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY, sinogram->getDataSinogramSize()*sizeof(float), NULL,&ret);
	printf("device_sinogram created\n");
	cl_mem device_sampling = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_sampling_opencl1), NULL,&ret);
	printf("device_sampling created\n");
	cl_mem device_constant = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_constante_opencl1), NULL,&ret);
    cl_mem device_alpha_beta = clCreateBuffer(context, CL_MEM_READ_ONLY, projectionSinogramNb * sizeof(cl_float2), NULL,&ret);
	printf("device_constant created\n");
    ret = clEnqueueWriteBuffer(queue, device_sinogram, CL_TRUE, 0, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_sampling, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_sampling, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_constant, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_constant, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_alpha_beta, CL_TRUE, 0, projectionSinogramNb * sizeof(cl_float2), alpha_beta, 0, NULL, NULL);

    cl_int kernel_status;
	// Create the program.


   

    
    //program = clCreateProgramWithSource(context, 1,(const char **)&source_str, NULL, &ret);
    program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&source_size, (const unsigned char **)&source_str, &kernel_status, &ret);
    CHECK_RET(ret, "source failed\n");

    if (ret != CL_SUCCESS) {
        puts("Could not create from binary");
        CLEANUP();
		exit(0);
    }
///////////////////////Buiding the program///////////////////////////////
    /*ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    ret = clFlush(queue);
    char *build_str;
    size_t logSize;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    build_str= new char[logSize + 1];
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize, build_str, NULL);
    build_str[logSize] = '\0';
    std::cout << build_str << std::endl;
    delete[] build_str;
    CHECK_RET(ret, "Failed to build program");
    printf("Program building done\n");*/


    // Creation de la fonction backprojection3D
    kernel = clCreateKernel(program, "backprojection3D", &ret);
    CHECK_RET(ret, "Failed to create kernel");
    printf("Kernel backprojection3D created\n");
    int argk = 0;
   
	ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_volume);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sinogram);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sampling);
    //ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_constant);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_alpha_beta);
    ret = clSetKernelArg(kernel, argk++, sizeof(float), &host_cst);

    size_t globalWorkItemSize[] = {256, 256, 256}; //The total size of 1 dimension of the work items. Here, 256*256*256
    //size_t workGroupSize = 256; // The size of one work group. Here, we have only one work-group
    
    //ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkItemSize, NULL, 0, NULL, NULL);
    double exe_time=0.0;
    for(int cpt = 0; cpt < 1; cpt++){
        ret = clEnqueueTask(queue, kernel, 1, &user_event, &kernel_event);

        if (ret != CL_SUCCESS) {
            printf("Error Enqueue: %d\n", ret);
            CLEANUP();
            exit(0);
        } else {
        printf("Kernel backprojection3D enqueue SUCCESS !\n");
        }

        // Lancement des kernels simultanément
        clSetUserEventStatus(user_event, CL_COMPLETE);
        
        clWaitForEvents(1, &kernel_event);

        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
        elapsed = (time_end - time_start);
        exe_time += elapsed;
        printf("Time kernels : %f s \n",elapsed/1000000000.0);
    }
    exe_time = exe_time/10;
    printf("Mean Time kernels : %.2f s \n",exe_time/1000000000.0);
    ret = clEnqueueReadBuffer(queue, device_volume, CL_TRUE, 0, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), host_volume, 0, NULL, NULL);
    
    estimatedVolume->setVolumeData((float *) host_volume);
	

	CLEANUP();  


}




template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::doBackProjection_GPU(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
{
	std::cout << "\tVI OpenCL BackProjection on GPU" << std::endl;
	TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>* plan;

	this->setVolume(estimatedVolume);
    
	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
    float uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
    float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();

	unsigned long int uSinogramPixelNb = sinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = sinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramNb = sinogram->getProjectionSinogramNb();

	unsigned long long int xThreadNb = this->getOCLBProjectionArchitecture()->getXThreadNb();
	unsigned long long int yThreadNb = this->getOCLBProjectionArchitecture()->getYThreadNb();
	unsigned long long int zThreadNb = this->getOCLBProjectionArchitecture()->getZThreadNb();

	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float yVolumePixelSize = this->getVolume()->getYVolumePixelSize();
	float zVolumePixelSize = this->getVolume()->getZVolumePixelSize();

	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();

	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	unsigned long long int xBlockNb = this->getOCLBProjectionArchitecture()->getXBlockNb();
	unsigned long long int yBlockNb = this->getOCLBProjectionArchitecture()->getYBlockNb();
	unsigned long long int zBlockNb = this->getOCLBProjectionArchitecture()->getZBlockNb();

	//CODE OPENCL PROVENANT DE TOMOX

    // Initialisation des objets OpenCL
    static cl_platform_id *platform_id = NULL;
    static cl_device_id device_id = NULL;
    static cl_context context = NULL;
    static cl_command_queue queue;
    static cl_kernel kernel;
    static cl_program program = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_event kernel_event;

    //Lecture du .aocx
    unsigned char *source_str;
    size_t source_size;
    FILE *fp;
    char fileName[255];
    sprintf(fileName,"%s/%s",getenv ("TOMO_GPI"),SOURCE_FILE); //"backprojection3D_kernel.aocx";
    fp = fopen(fileName, "rb");
    if (!fp) {
    	   fprintf(stderr, "Failed to load kernel.\n");
         exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (unsigned char*) malloc(source_size * sizeof(unsigned char));
    if (fread(source_str, 1, source_size, fp) == 0) {
        puts("Could not read source file");
        exit(-1);
    }
    printf("Taille du binaire : %lu bytes\n", source_size);
    fclose(fp);

  
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

   // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_RET(ret, "Create Context");

    cl_event user_event = clCreateUserEvent(context, &ret);

    double elapsed = 0;
    cl_ulong time_start, time_end;

    // Obligation de recopie de certaines donnees : OpenCL ne gere pas la recopie de pointeurs de pointeurs	
	type_struct_constante_opencl1 * host_constant = (type_struct_constante_opencl1 * ) malloc(sizeof(type_struct_constante_opencl1));
	type_struct_sampling_opencl1* host_sampling = (type_struct_sampling_opencl1*) malloc (sizeof(type_struct_sampling_opencl1));

	float * host_volume = (float *)estimatedVolume->getVolumeData();
	printf("Taille host_volume : %d * sizeof(float)\n",estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb());

    //float * host_sinogram = (float *) malloc(sizeof(float)*sinogram->getDataSinogramSize());
    //host_sinogram = (float *) sinogram->getDataSinogram();
    float * host_sinogram = (float *)sinogram->getDataSinogram();
	printf("Taille host_sinogram : %d * sizeof(float)\n",sinogram->getDataSinogramSize());

    
//for(uint l=0; l<sinogram->getDataSinogramSize(); l++){
  //  printf("%f\t", host_sinogram[l]);
//}
 
	cl_ulong3 host_vol_dims;
	cl_ulong3 host_sin_dims;

	// Division effectuee du cote hote
	float host_cst = xVolumePixelSize * (fdd / uDetectorPixelSize);
	printf("Valeur host_const : %f\n", host_cst);
    

	host_vol_dims.x=xVolumePixelNb;
	host_vol_dims.y=yVolumePixelNb;
	host_vol_dims.z=zVolumePixelNb;

	host_sin_dims.x=uSinogramPixelNb;
	host_sin_dims.y=vSinogramPixelNb;
	host_sin_dims.z=projectionSinogramNb;

	host_sampling->delta_xn=xVolumePixelSize;
	host_sampling->N_un=uDetectorPixelNb;
	host_sampling->N_vn=vDetectorPixelNb;

	host_sampling->xn_0=xVolumeCenterPixel;
	host_sampling->yn_0=yVolumeCenterPixel;
	host_sampling->zn_0=zVolumeCenterPixel;

	host_constant->gamma_wn=fod;
	host_constant->gamma_vn=1.0/this->getGammaIOcylinderC(); //(sampling->D*sampling->delta_zn)/(sampling->delta_vn);
	host_constant->delta_un=uDetectorPixelSize;
	host_constant->un_0=uDetectorCenterPixel;
	host_constant->vn_0=vDetectorCenterPixel;
	host_constant->gamma_D=(vDetectorPixelSize*fod)/(fdd*zVolumePixelSize);//(sampling->delta_vn*sampling->R)/(sampling->D*sampling->delta_zn);
	host_constant->D=fdd;

    printf("test_value: %f\n", host_constant->vn_0);

    cl_float2 * alpha_beta = (cl_float2 *) malloc(sizeof(cl_float2)*projectionSinogramNb);

    for (int p=0; p<projectionSinogramNb;p++){
	    //host_constant->alpha_wn[p] = this->getAlphaIOcylinderC()[p];
	    //host_constant->beta_wn[p] = this->getBetaIOcylinderC()[p];
        alpha_beta[p].s0=this->getAlphaIOcylinderC()[p]; 
        alpha_beta[p].s1=this->getBetaIOcylinderC()[p];
       // printf("%f\t", host_constant->alpha_wn[p]);
    }


    //NICOLAS  : memcpy non nécessaire à mon avis ???
    //memcpy(host_volume,volume->D.f,volume->N_xn*volume->N_yn*volume->N_zn*sizeof(float));
    //memcpy(host_sinogram,sinogram->D.f,sinogram->N_phi*sinogram->N_un*sinogram->N_vn*sizeof(float));

/*	printf("After memcpy1\n");

	for (int i=0;i<4;i++){
		printf("host_volume[%d]=%f\n",i,host_volume[i]);
	}*/
     
    //queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem device_volume = clCreateBuffer(context, CL_MEM_READ_WRITE, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), NULL,&ret);
	printf("device_volume created\n");
	//cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram,&ret);
    cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY, sinogram->getDataSinogramSize()*sizeof(float), NULL,&ret);
	printf("device_sinogram created\n");
	cl_mem device_sampling = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_sampling_opencl1), NULL,&ret);
	printf("device_sampling created\n");
	cl_mem device_constant = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_constante_opencl1), NULL,&ret);
    cl_mem device_alpha_beta = clCreateBuffer(context, CL_MEM_READ_ONLY, projectionSinogramNb * sizeof(cl_float2), NULL,&ret);
	printf("device_constant created\n");
    ret = clEnqueueWriteBuffer(queue, device_sinogram, CL_TRUE, 0, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_sampling, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_sampling, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_constant, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_constant, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_alpha_beta, CL_TRUE, 0, projectionSinogramNb * sizeof(cl_float2), alpha_beta, 0, NULL, NULL);

    cl_int kernel_status;
	// Create the program.


   

    
    program = clCreateProgramWithSource(context, 1,(const char **)&source_str, NULL, &ret);
    //program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&source_size, (const unsigned char **)&source_str, &kernel_status, &ret);
    CHECK_RET(ret, "source failed\n");

    if (ret != CL_SUCCESS) {
        puts("Could not create from binary");
        CLEANUP();
		exit(0);
    }
///////////////////////Buiding the program///////////////////////////////
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    ret = clFlush(queue);
    char *build_str;
    size_t logSize;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    build_str= new char[logSize + 1];
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize, build_str, NULL);
    build_str[logSize] = '\0';
    std::cout << build_str << std::endl;
    delete[] build_str;
    CHECK_RET(ret, "Failed to build program");
    printf("Program building done\n");


    // Creation de la fonction backprojection3D
    kernel = clCreateKernel(program, "backprojection3D_NDR", &ret);
    CHECK_RET(ret, "Failed to create kernel");
    printf("Kernel backprojection3D created\n");
    int argk = 0;
    
	ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_volume);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sinogram);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sampling);
    //ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_constant);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_alpha_beta);
    ret = clSetKernelArg(kernel, argk++, sizeof(float), &host_cst);

    size_t globalWorkItemSize[] = {256, 256, 256}; //The total size of 1 dimension of the work items. Here, 256*256*256
    //size_t workGroupSize = 256; // The size of one work group. Here, we have only one work-group
    
    ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkItemSize, NULL, 0, NULL, &kernel_event);

    //ret = clEnqueueTask(queue, kernel, 1, &user_event, &kernel_event);

    if (ret != CL_SUCCESS) {
        printf("Error Enqueue: %d\n", ret);
        CLEANUP();
        exit(0);
    } else {
       printf("Kernel backprojection3D enqueue SUCCESS !\n");
    }

    // Lancement des kernels simultanément
    clSetUserEventStatus(user_event, CL_COMPLETE);
    
    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    elapsed = (time_end - time_start);
    printf("Time kernels : %f\n",elapsed/1000000000.0);

    ret = clEnqueueReadBuffer(queue, device_volume, CL_TRUE, 0, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), host_volume, 0, NULL, NULL);
    
    estimatedVolume->setVolumeData(host_volume);

    //estimatedVolume->saveVolume("daouda.v");
    //memcpy(volume->D.f,host_volume,volume->N_zn*volume->N_xn*volume->N_yn*sizeof(float));
	

	CLEANUP();  


}


template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::doBackProjection_CPU(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
{
	std::cout << "\tVI OpenCL BackProjection " << std::endl;
	TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>* plan;

	this->setVolume(estimatedVolume);
    
	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
    float uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
    float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();

	unsigned long int uSinogramPixelNb = sinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = sinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramNb = sinogram->getProjectionSinogramNb();

	unsigned long long int xThreadNb = this->getOCLBProjectionArchitecture()->getXThreadNb();
	unsigned long long int yThreadNb = this->getOCLBProjectionArchitecture()->getYThreadNb();
	unsigned long long int zThreadNb = this->getOCLBProjectionArchitecture()->getZThreadNb();

	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float yVolumePixelSize = this->getVolume()->getYVolumePixelSize();
	float zVolumePixelSize = this->getVolume()->getZVolumePixelSize();

	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();

	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	unsigned long long int xBlockNb = this->getOCLBProjectionArchitecture()->getXBlockNb();
	unsigned long long int yBlockNb = this->getOCLBProjectionArchitecture()->getYBlockNb();
	unsigned long long int zBlockNb = this->getOCLBProjectionArchitecture()->getZBlockNb();

	//CODE OPENCL PROVENANT DE TOMOX

    // Initialisation des objets OpenCL
    static cl_platform_id *platform_id = NULL;
    static cl_device_id device_id = NULL;
    static cl_context context = NULL;
    static cl_command_queue queue;
    static cl_kernel kernel;
    static cl_kernel kernel1;
    static cl_program program = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_event kernel_event;
    cl_event kernel_event1;

    //Lecture du .aocx
    unsigned char *source_str;
    size_t source_size;
    FILE *fp;
    char fileName[255];
     sprintf(fileName,"%s/%s",getenv ("TOMO_GPI"),SOURCE_FILE); //"backprojection3D_kernel.aocx";
    fp = fopen(fileName, "rb");
    if (!fp) {
    	   fprintf(stderr, "Failed to load kernel.\n");
         exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (unsigned char*) malloc(source_size * sizeof(unsigned char));
    if (fread(source_str, 1, source_size, fp) == 0) {
        puts("Could not read source file");
        exit(-1);
    }
    printf("Taille du binaire : %lu bytes\n", source_size);
    fclose(fp);

  
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    ret = clGetDeviceIDs(platform_id[1], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

   // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_RET(ret, "Create Context");

    cl_event user_event = clCreateUserEvent(context, &ret);

    double elapsed = 0;
    cl_ulong time_start, time_end;

    // Obligation de recopie de certaines donnees : OpenCL ne gere pas la recopie de pointeurs de pointeurs	
	type_struct_constante_opencl1 * host_constant = (type_struct_constante_opencl1 * ) malloc(sizeof(type_struct_constante_opencl1));
	type_struct_sampling_opencl1* host_sampling = (type_struct_sampling_opencl1*) malloc (sizeof(type_struct_sampling_opencl1));

	float * host_volume = (float *)estimatedVolume->getVolumeData();
    float * host_volume1 = (float *)estimatedVolume->getVolumeData();
	printf("Taille host_volume : %d * sizeof(float)\n",estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb());

    //float * host_sinogram = (float *) malloc(sizeof(float)*sinogram->getDataSinogramSize());
    //host_sinogram = (float *) sinogram->getDataSinogram();
    float * host_sinogram = (float *)sinogram->getDataSinogram();
	printf("Taille host_sinogram : %d * sizeof(float)\n",sinogram->getDataSinogramSize());

    
//for(uint l=0; l<sinogram->getDataSinogramSize(); l++){
  //  printf("%f\t", host_sinogram[l]);
//}
 
	cl_ulong3 host_vol_dims;
	cl_ulong3 host_sin_dims;

	// Division effectuee du cote hote
	float host_cst = xVolumePixelSize * (fdd / uDetectorPixelSize);
	printf("Valeur host_const : %f\n", host_cst);
    

	host_vol_dims.x=xVolumePixelNb;
	host_vol_dims.y=yVolumePixelNb;
	host_vol_dims.z=zVolumePixelNb;

	host_sin_dims.x=uSinogramPixelNb;
	host_sin_dims.y=vSinogramPixelNb;
	host_sin_dims.z=projectionSinogramNb;

	host_sampling->delta_xn=xVolumePixelSize;
	host_sampling->N_un=uDetectorPixelNb;
	host_sampling->N_vn=vDetectorPixelNb;

	host_sampling->xn_0=xVolumeCenterPixel;
	host_sampling->yn_0=yVolumeCenterPixel;
	host_sampling->zn_0=zVolumeCenterPixel;

	host_constant->gamma_wn=fod;
	host_constant->gamma_vn=1.0/this->getGammaIOcylinderC(); //(sampling->D*sampling->delta_zn)/(sampling->delta_vn);
	host_constant->delta_un=uDetectorPixelSize;
	host_constant->un_0=uDetectorCenterPixel;
	host_constant->vn_0=vDetectorCenterPixel;
	host_constant->gamma_D=(vDetectorPixelSize*fod)/(fdd*zVolumePixelSize);//(sampling->delta_vn*sampling->R)/(sampling->D*sampling->delta_zn);
	host_constant->D=fdd;

    printf("test_value: %f\n", host_constant->vn_0);

    cl_float2 * alpha_beta = (cl_float2 *) malloc(sizeof(cl_float2)*projectionSinogramNb);

    for (int p=0; p<projectionSinogramNb;p++){
	    //host_constant->alpha_wn[p] = this->getAlphaIOcylinderC()[p];
	    //host_constant->beta_wn[p] = this->getBetaIOcylinderC()[p];
        alpha_beta[p].s0=this->getAlphaIOcylinderC()[p]; 
        alpha_beta[p].s1=this->getBetaIOcylinderC()[p];
       // printf("%f\t", host_constant->alpha_wn[p]);
    }


    //NICOLAS  : memcpy non nécessaire à mon avis ???
    //memcpy(host_volume,volume->D.f,volume->N_xn*volume->N_yn*volume->N_zn*sizeof(float));
    //memcpy(host_sinogram,sinogram->D.f,sinogram->N_phi*sinogram->N_un*sinogram->N_vn*sizeof(float));

	/*printf("After memcpy1\n");

	for (int i=0;i<4;i++){
		printf("host_volume[%d]=%f\n",i,host_volume[i]);
	}*/
     
    //queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem device_volume = clCreateBuffer(context, CL_MEM_READ_WRITE, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), NULL,&ret);
    printf("device_volume created\n");
	//cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram,&ret);
    cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY, sinogram->getDataSinogramSize()*sizeof(float), NULL,&ret);
	printf("device_sinogram created\n");
	cl_mem device_sampling = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_sampling_opencl1), NULL,&ret);
	printf("device_sampling created\n");
	cl_mem device_constant = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_constante_opencl1), NULL,&ret);
    cl_mem device_alpha_beta = clCreateBuffer(context, CL_MEM_READ_ONLY, projectionSinogramNb * sizeof(cl_float2), NULL,&ret);
	printf("device_constant created\n");
    ret = clEnqueueWriteBuffer(queue, device_sinogram, CL_TRUE, 0, sinogram->getDataSinogramSize()*sizeof(float), host_sinogram, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_sampling, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_sampling, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_constant, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_constant, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_alpha_beta, CL_TRUE, 0, projectionSinogramNb * sizeof(cl_float2), alpha_beta, 0, NULL, NULL);

    cl_int kernel_status;
	// Create the program.


   

    
    program = clCreateProgramWithSource(context, 1,(const char **)&source_str, NULL, &ret);
    //program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&source_size, (const unsigned char **)&source_str, &kernel_status, &ret);
    CHECK_RET(ret, "source failed\n");

    if (ret != CL_SUCCESS) {
        puts("Could not create from binary");
        CLEANUP();
		exit(0);
    }
///////////////////////Buiding the program///////////////////////////////
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    ret = clFlush(queue);
    char *build_str;
    size_t logSize;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    build_str= new char[logSize + 1];
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize, build_str, NULL);
    build_str[logSize] = '\0';
    std::cout << build_str << std::endl;
    delete[] build_str;
    CHECK_RET(ret, "Failed to build program");
    printf("Program building done\n");


    // Creation de la fonction backprojection3D
    kernel = clCreateKernel(program, "backprojection3D", &ret);
    CHECK_RET(ret, "Failed to create kernel");
    printf("Kernel backprojection3D created\n");
    // kernel1 = clCreateKernel(program, "backprojection3D_SWI", &ret);
    // CHECK_RET(ret, "Failed to create kernel1");
    int argk = 0;
    
	ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_volume);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sinogram);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sampling);
    //ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_constant);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_alpha_beta);
    ret = clSetKernelArg(kernel, argk++, sizeof(float), &host_cst);


    size_t globalWorkItemSize[] = {256, 256, 256}; //The total size of 1 dimension of the work items. Here, 256*256*256
    //size_t workGroupSize = 256; // The size of one work group. Here, we have only one work-group
    
    //ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkItemSize, NULL, 0, NULL, &kernel_event);

    ret = clEnqueueTask(queue, kernel, 1, &user_event, &kernel_event);

    if (ret != CL_SUCCESS) {
        printf("Error Enqueue: %d\n", ret);
        CLEANUP();
        exit(0);
    } else {
       printf("Kernel backprojection3D enqueue SUCCESS !\n");
    }

    // Lancement des kernels simultanément
    clSetUserEventStatus(user_event, CL_COMPLETE);
    
    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    elapsed = (time_end - time_start);
    printf("Time kernels : %f\n",elapsed/1000000000.0);

    ret = clEnqueueReadBuffer(queue, device_volume, CL_TRUE, 0, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), host_volume, 0, NULL, NULL);
    

    //estimatedVolume->setVolumeData(host_volume);

    //estimatedVolume->saveVolume("daouda.v");
    //memcpy(volume->D.f,host_volume,volume->N_zn*volume->N_xn*volume->N_yn*sizeof(float));
	

	CLEANUP();  


}





template <typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::EnableP2P(){}

template <typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::CopyConstant(){}

template <typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::DisableP2P(){}


template<typename T>
OCLBProjectionArchitecture* VIBackProjector_compute_OCL_mem_CPU<T>::getOCLBProjectionArchitecture() const
{
	return this->oclbackprojectionArchitecture;
}

template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::setOCLBProjectionArchitecture(OCLBProjectionArchitecture*  oclbackprojectionArchitecture)
{
	this->oclbackprojectionArchitecture =  oclbackprojectionArchitecture;
}

#include "BackProjector_instances_CPU.cu"
