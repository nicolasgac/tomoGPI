/*
 * VIBackProjector_compute_OCL_mem_CPU_real.cu
 *
 *      Author: diakite
 */

#include "BackProjector_CPU.cuh"

#define SOURCE_FILE "/home/daoudadiakite/Documents/tomo_gpi/TomoBayes/src/TomoBayes_lib/src/OCL/hello_world.cl"


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




template<typename T>
VIBackProjector_compute_OCL_mem_CPU<T>::VIBackProjector_compute_OCL_mem_CPU(Acquisition* acquisition, Detector* detector, OCLBProjectionArchitecture *oclbackprojectionArchitecture, Volume_CPU<T>* volume,char fdk) : BackProjector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector,volume,fdk){
    this->setOCLBProjectionArchitecture(oclbackprojectionArchitecture);
}

template<typename T>
VIBackProjector_compute_OCL_mem_CPU<T>::~VIBackProjector_compute_OCL_mem_CPU(){}



template<typename T>
void VIBackProjector_compute_OCL_mem_CPU<T>::doBackProjection(Volume_CPU<T>* estimatedVolume,Sinogram3D_CPU<T>* sinogram)
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
    char fileName[] = SOURCE_FILE;//"backprojection3D_kernel.aocx";
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

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_RET(ret, "Create Context");

    cl_event user_event = clCreateUserEvent(context, &ret);

    double elapsed = 0;
    cl_ulong time_start, time_end;

    // Obligation de recopie de certaines donnees : OpenCL ne gere pas la recopie de pointeurs de pointeurs	
	type_struct_constante_opencl * host_constant = (type_struct_constante_opencl * ) malloc(sizeof(type_struct_constante_opencl));
	type_struct_sampling_opencl* host_sampling = (type_struct_sampling_opencl*) malloc (sizeof(type_struct_sampling_opencl));

	float * host_volume = (float *)estimatedVolume->getVolumeData();
	printf("Taille host_volume : %d * sizeof(float)\n",estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb());

	float * host_sinogram = (float *)sinogram->getDataSinogramSize();
	printf("Taille host_sinogram : %d * sizeof(float)\n",sinogram->getDataSinogramSize());

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

    for (int p;p<projectionSinogramNb;p++){
	    host_constant->alpha_wn[p] = this->getAlphaIOcylinderC()[p];
	    host_constant->beta_wn[p] = this->getBetaIOcylinderC()[p];
    }

	printf("Before memcpy\n");

    //NICOLAS  : memcpy non nécessaire à mon avis ???
    //memcpy(host_volume,volume->D.f,volume->N_xn*volume->N_yn*volume->N_zn*sizeof(float));
    //memcpy(host_sinogram,sinogram->D.f,sinogram->N_phi*sinogram->N_un*sinogram->N_vn*sizeof(float));

	/*printf("After memcpy1\n");

	for (int i=0;i<4;i++){
		printf("host_volume[%d]=%f\n",i,host_volume[i]);
	}
     
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem device_volume = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, estimatedVolume->getXVolumePixelNb()*estimatedVolume->getYVolumePixelNb()*estimatedVolume->getZVolumePixelNb()*sizeof(float), host_volume,&ret);
	printf("device_volume created\n");
	cl_mem device_sinogram = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sinogram->getDataSinogramSize()*sizeof(float), host_sinogram,&ret);
	printf("device_sinogram created\n");
	cl_mem device_sampling = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(type_struct_sampling_opencl), host_sampling,&ret);
	printf("device_sampling created\n");
	cl_mem device_constant = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(type_struct_constante_opencl), host_constant,&ret);
	printf("device_constant created\n");*/


    cl_int kernel_status;
	// Create the program.
    //program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&source_size, (const unsigned char **)&source_str, &kernel_status, &ret);
    //const char * code_kernel = Read_Source_File(SOURCE_FILE); 

   

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    program = clCreateProgramWithSource(context, 1,(const char **)&source_str, NULL, &ret);
    CHECK_RET(ret, "source failed\n");

    if (ret != CL_SUCCESS) {
        puts("Could not create from binary");
        CLEANUP();
		exit(0);
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    ret = clFlush(queue);
    //CHECK_RET(ret, "Failed to build program");
    printf("Program building done\n");


    // Creation de la fonction backprojection3D
    kernel = clCreateKernel(program, "hello_world", &ret);
    CHECK_RET(ret, "Failed to create kernel");
    printf("Kernel hello_world created\n");

    //ret = clEnqueueWriteBuffer(queue, device_sinogram, CL_TRUE, 0, sinogram->N_phi*sinogram->N_un*sinogram->N_vn*sizeof(float), ////////host_sinogram, 0, NULL, NULL);
	//if (ret != CL_SUCCESS)
    //{
     //   printf("Error: Failed to write device_sinogram!\n");
      //  exit(1);
    //}

    //ret = clEnqueueWriteBuffer(queue, device_sampling, CL_TRUE, 0, sizeof(type_struct_sampling_opencl), host_sampling, 0, NULL, NULL);
	//if (ret != CL_SUCCESS)
    //{
     //   printf("Error: Failed to write device_sinogram!\n");
      //  exit(1);
    //}

    //ret = clEnqueueWriteBuffer(queue, device_constant, CL_TRUE, 0, sizeof(type_struct_constante_opencl), host_constant, 0, NULL, NULL);
	//if (ret != CL_SUCCESS)
    //{
     //   printf("Error: Failed to write device_sinogram!\n");
      //  exit(1);
    //}
for(int i=0; i<10; i++){
    int argk = 0; int thread_to_test=i;
    ret = clSetKernelArg(kernel, argk++, sizeof(int), &thread_to_test);


    int work_item_size = 10;
    size_t globalWorkItemSize = work_item_size; //The total size of 1 dimension of the work items. Here, we set 10 work-items
    size_t workGroupSize = work_item_size; // The size of one work group. Here, we have only one work-group
    
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize, 0, NULL, NULL);

    CHECK_RET(ret, "Failed to create command queue");
}
	/*ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_volume);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sinogram);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_sampling);
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), &device_constant);
    ret = clSetKernelArg(kernel, argk++, sizeof(float), &host_cst);

    ret = clEnqueueTask(queue, kernel, 1, &user_event, &kernel_event);*/

    if (ret != CL_SUCCESS) {
        printf("Error Enqueue: %d\n", ret);
        CLEANUP();
        exit(0);
    } else {
       printf("Kernel backprojection3D enqueue SUCCESS !\n");
    }

    // Lancement des kernels simultanément
    clSetUserEventStatus(user_event, CL_COMPLETE);
    
    //clWaitForEvents(1, &kernel_event);

    //clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    //clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    elapsed = (time_end - time_start);
    printf("Time kernels : %f\n",elapsed/1000000000.0);

    //ret = clEnqueueReadBuffer(queue, device_volume, CL_TRUE, 0, volume->N_zn*volume->N_xn*volume->N_yn*sizeof(float), host_volume, 0, NULL, NULL);
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
