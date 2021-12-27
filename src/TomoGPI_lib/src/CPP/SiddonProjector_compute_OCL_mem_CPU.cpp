/*
 * Author: gac, diakite
 */

#include "Projector_CPU.cuh"

#define SOURCE_FILE "TomoBayes/src/TomoBayes_lib/src/OCL/projection3D_Siddon.cl"

//#include "opencl_compat.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif
#include "AOCLUtils/aocl_utils.h"

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

#define PHI_MAX 1000
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


/* RegularSamplingProjector definition */
template <typename T>
SiddonProjector_compute_OCL_mem_CPU<T>::SiddonProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector, OCLProjectionArchitecture *oclprojectionArchitecture, Volume_CPU<T>* volume) : Projector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector, volume){
    this->setOCLProjectionArchitecture(oclprojectionArchitecture);
}

template <typename T>
SiddonProjector_compute_OCL_mem_CPU<T>::~SiddonProjector_compute_OCL_mem_CPU(){}

template <typename T>
void SiddonProjector_compute_OCL_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume)
{
   std::cout << "\tOCL Siddon Projection " << std::endl;
   //TGPUplan_retro<Volume_CPU, Sinogram3D_CPU, T>* plan;

	this->setVolume(volume);
    // data
	T* d_volume=this->getVolume()->getVolumeData();
	T* sinogram_g=estimatedSinogram->getDataSinogram();
    
	float fdd = this->getAcquisition()->getFocusDetectorDistance();
	float fod = this->getAcquisition()->getFocusObjectDistance();
    float uDetectorPixelNb = this->getDetector()->getUDetectorPixelNb();
	float uDetectorPixelSize = this->getDetector()->getUDetectorPixelSize();
	float uDetectorCenterPixel = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorPixelNb = this->getDetector()->getVDetectorPixelNb();
    float vDetectorPixelSize = this->getDetector()->getVDetectorPixelSize();
    float uDetectorCenterPixel_GPU = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorCenterPixel = this->getDetector()->getVDetectorCenterPixel();
    float uDetectorPixelSize_GPU = this->getDetector()->getUDetectorPixelSize();
	float vDetectorPixelSize_GPU = this->getDetector()->getVDetectorPixelSize();

	unsigned long int uSinogramPixelNb = estimatedSinogram->getUSinogramPixelNb();
	unsigned long int vSinogramPixelNb = estimatedSinogram->getVSinogramPixelNb();
	unsigned long int projectionSinogramNb = estimatedSinogram->getProjectionSinogramNb();

	unsigned long long int xThreadNb = this->getOCLProjectionArchitecture()->getXThreadNb();
	unsigned long long int yThreadNb = this->getOCLProjectionArchitecture()->getYThreadNb();
	unsigned long long int zThreadNb = this->getOCLProjectionArchitecture()->getZThreadNb();

	float xVolumePixelSize = this->getVolume()->getXVolumePixelSize();
	float yVolumePixelSize = this->getVolume()->getYVolumePixelSize();
	float zVolumePixelSize = this->getVolume()->getZVolumePixelSize();

	float xVolumeCenterPixel = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel = this->getVolume()->getZVolumeCenterPixel();

	unsigned long int xVolumePixelNb = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb = this->getVolume()->getZVolumePixelNb();

	unsigned long long int xBlockNb = this->getOCLProjectionArchitecture()->getXBlockNb();
	unsigned long long int yBlockNb = this->getOCLProjectionArchitecture()->getYBlockNb();
	unsigned long long int zBlockNb = this->getOCLProjectionArchitecture()->getZBlockNb();

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
    ret = clGetDeviceIDs(platform_id[1], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

   // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_RET(ret, "Create Context");

    cl_event user_event = clCreateUserEvent(context, &ret);

    double elapsed = 0;
    cl_ulong time_start, time_end;

    // Obligation de recopie de certaines donnees : OpenCL ne gere pas la recopie de pointeurs de pointeurs	
	//type_struct_constante_opencl1 * host_constant = (type_struct_constante_opencl1 * ) malloc(sizeof(type_struct_constante_opencl1));
	//type_struct_sampling_opencl1* host_sampling = (type_struct_sampling_opencl1*) malloc (sizeof(type_struct_sampling_opencl1));

    
    //void * host_volume = NULL; //(cl_float*)alignedMalloc(volume->N_xn*volume->N_yn*volume->N_zn*sizeof(float));
	//posix_memalign ((void **)&host_volume, AOCL_ALIGNMENT, volume->getXVolumePixelNb()*volume->getYVolumePixelNb()*volume->getZVolumePixelNb()*sizeof(float));
    
	//float * host_volume = (float *)volume->getVolumeData();
	//printf("Taille host_volume : %d * sizeof(float)\n",volume->getXVolumePixelNb()*volume->getYVolumePixelNb()*volume->getZVolumePixelNb());

    
    //void * host_sinogram = NULL; //(cl_float*)alignedMalloc(estimatedSinogram->N_phi*estimatedSinogram->N_un*estimatedSinogram->N_vn*sizeof(float));
	//posix_memalign ((void **)&host_sinogram, AOCL_ALIGNMENT, estimatedSinogram->getDataSinogramSize()*sizeof(float));

    //memcpy(host_volume, d_volume,volume->getXVolumePixelNb()*volume->getYVolumePixelNb()*volume->getZVolumePixelNb()*sizeof(float));
   	//memcpy(host_sinogram,sinogram_g,estimatedSinogram->getDataSinogramSize()*sizeof(float));
    
    
    float * host_alpha = NULL;
    posix_memalign ((void **)&host_alpha, AOCL_ALIGNMENT, 256 * sizeof(float));
    
    float * host_beta = NULL;
    posix_memalign ((void **)&host_beta, AOCL_ALIGNMENT, 256 * sizeof(float));



    //float * host_sinogram = (float *)estimatedSinogram->getDataSinogram();
	//printf("Taille host_sinogram : %d * sizeof(float)\n",sinogram->getDataSinogramSize());
 
	cl_ulong3 host_vol_dims;
	cl_ulong3 host_sin_dims;
	cl_float8 host_constant;
	cl_float2 host_delta;

    

	host_vol_dims.x=xVolumePixelNb;
	host_vol_dims.y=yVolumePixelNb;
	host_vol_dims.z=zVolumePixelNb;

	host_sin_dims.x=uSinogramPixelNb;
	host_sin_dims.y=vSinogramPixelNb;
	host_sin_dims.z=projectionSinogramNb;

	// host_sampling->delta_xn=xVolumePixelSize;
	// host_sampling->N_un=uDetectorPixelNb;
	// host_sampling->N_vn=vDetectorPixelNb;

    host_constant.s0=fdd;
    host_constant.s1=fod;
	host_constant.s2=xVolumeCenterPixel; //host_sampling->xn_0
	host_constant.s3=yVolumeCenterPixel;//host_sampling->yn_0
	host_constant.s4=zVolumeCenterPixel; //host_sampling->zn_0
    host_constant.s5=xVolumePixelSize;
	host_constant.s6=uDetectorCenterPixel;
	host_constant.s7=vDetectorCenterPixel;
    //host_constant->gamma_vn=1.0/this->getGammaIOcylinderC(); //(sampling->D*sampling->delta_zn)/(sampling->delta_vn);
	//host_constant->gamma_D=(vDetectorPixelSize*fod)/(fdd*zVolumePixelSize);//(sampling->delta_vn*sampling->R)/(sampling->D*sampling->delta_zn);
	 

    host_delta.s0=uDetectorPixelSize_GPU;
    host_delta.s1=vDetectorPixelSize_GPU;
    
	// host_constant->gamma_wn=fod;
	// host_constant->gamma_vn=1.0/this->getGammaIOcylinderC(); //(sampling->D*sampling->delta_zn)/(sampling->delta_vn);
	// host_constant->delta_un=uDetectorPixelSize;
	// host_constant->un_0=uDetectorCenterPixel;
	// host_constant->vn_0=vDetectorCenterPixel;
	// host_constant->gamma_D=(vDetectorPixelSize*fod)/(fdd*zVolumePixelSize);//(sampling->delta_vn*sampling->R)/(sampling->D*sampling->delta_zn);
	// host_constant->D=fdd;

     for (int p=0; p<projectionSinogramNb;p++){
	    
        host_alpha[p]=this->getAlphaIOcylinderC()[p]; // vector first element is alpha
        host_beta[p]=this->getBetaIOcylinderC()[p];  // vector second element is beta
      
    }

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem device_volume = clCreateBuffer(context, CL_MEM_READ_ONLY, volume->getXVolumePixelNb()*volume->getYVolumePixelNb()*volume->getZVolumePixelNb()*sizeof(float), NULL,&ret);
	printf("device_volume created\n");
    cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_WRITE_ONLY, estimatedSinogram->getDataSinogramSize()*sizeof(float), NULL,&ret);
	printf("device_sinogram created\n");
	cl_mem device_sampling = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_sampling_opencl1), NULL,&ret);
	printf("device_sampling created\n");
	cl_mem device_constant = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(type_struct_constante_opencl1), NULL,&ret);
    cl_mem device_alpha = clCreateBuffer(context, CL_MEM_READ_ONLY, projectionSinogramNb * sizeof(float), NULL,&ret);
     cl_mem device_beta = clCreateBuffer(context, CL_MEM_READ_ONLY, projectionSinogramNb * sizeof(float), NULL,&ret);
	printf("device_constant created\n");
    ret = clEnqueueWriteBuffer(queue, device_volume, CL_TRUE, 0, volume->getXVolumePixelNb()*volume->getYVolumePixelNb()*volume->getZVolumePixelNb()*sizeof(float), d_volume, 0, NULL, NULL);
    //ret = clEnqueueWriteBuffer(queue, device_sampling, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_sampling, 0, NULL, NULL);
    //ret = clEnqueueWriteBuffer(queue, device_constant, CL_TRUE, 0, sizeof(type_struct_sampling_opencl1), host_constant, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_alpha, CL_TRUE, 0, projectionSinogramNb * sizeof(float), host_alpha, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, device_beta, CL_TRUE, 0, projectionSinogramNb * sizeof(float), host_beta, 0, NULL, NULL);

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
    // Creation de la fonction projection3D
    kernel = clCreateKernel(program, "projection3D", &ret);
    CHECK_RET(ret, "Could not create kernel");
    printf("Kernel projection3D created\n");

    int argk = 0;
	
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_volume);
    CHECK_RET(ret, "Failed to set argument 0");
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sinogram);
    CHECK_RET(ret, "Failed to set argument 1");
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_alpha);
    CHECK_RET(ret, "Failed to set argument 3");
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_beta);
    CHECK_RET(ret, "Failed to set argument 4");
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_float8), &host_constant);
    CHECK_RET(ret, "Failed to set argument 5");
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_float2), &host_delta);
    CHECK_RET(ret, "Failed to set argument 6");
    

	size_t global[] = {256,256,256};
	size_t local[] = {64,1,1};
	

	ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 1, &user_event, &kernel_event);
    //ret = clEnqueueTask(queue, kernel, 1, &user_event, &kernel_event);

    if (ret != CL_SUCCESS) {
        printf("Error Enqueue: %d\n", ret);
        CLEANUP();
        exit(0);
    } else {
       printf("Kernel projection3D enqueue SUCCESS !\n");
    }

    // Lancement des kernels simultanément
    clSetUserEventStatus(user_event, CL_COMPLETE);
    
    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    elapsed = (time_end - time_start);
    printf("Time kernels : %f\n",elapsed/1000000000.0);

     ret = clEnqueueReadBuffer(queue, device_sinogram, CL_TRUE, 0,  estimatedSinogram->getDataSinogramSize()*sizeof(float), sinogram_g, 0, NULL, NULL);
    
    //volume->setVolumeData((float *) host_volume);
	

	CLEANUP();  

}


template <typename T>
void SiddonProjector_compute_OCL_mem_CPU<T>::EnableP2P(){}

template <typename T>
void SiddonProjector_compute_OCL_mem_CPU<T>::DisableP2P(){}

template<typename T>
OCLProjectionArchitecture* SiddonProjector_compute_OCL_mem_CPU<T>::getOCLProjectionArchitecture() const
{
	return this->oclprojectionArchitecture;
}

template<typename T>
void SiddonProjector_compute_OCL_mem_CPU<T>::setOCLProjectionArchitecture(OCLProjectionArchitecture*  oclprojectionArchitecture)
{
	this->oclprojectionArchitecture =  oclprojectionArchitecture;
}

#include "Projector_instances_CPU.cu"