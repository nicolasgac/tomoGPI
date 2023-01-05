# TomoGPI benchmark

###  Building the project  ####
To build project  
	cd build/  
	cmake ..  
	make  

### Running the project ####
To run the exectutable  

	- cd Data3D_0256/  
	- TomoGPI_exe 256 --pair=RSVI --op=back --compute=OCL --architecture=FPGA  
	
	- "256" is the data size width N (N³ for 3D volume)  
	- "TomoGPI_exe" is the executable  
	- "pair" is the pair of projector/backprojector to run in this case RSVI voxel-driven interpolation (Other possible option "SIDDONVI")  
	- "op" is the operator "proj" for projector and "back" for backprojector  
	- "compute" option is for the target language "C" for language C or "CUDA" for CUDA on NVIDIA GPU or "OCL" for OpenCL or CUDA_OCL for proj on GPU and backproj on FPGA  
	- "architecture" for the target architecture CPU, GPU or FPGA.  

##### The executable should be copied in the Data3D_0256/ or simply added to the path

##### The OpenCL kernels are in the src/TomoGPI_lib/src/OCL/ for the 3D backprojector
##### There are one for for Arria 10 device and two files for Stratix 10 device (single and multi kernels)



##### For each kernel one can tune the block size and shape and fixes the projection data size for the local memory 

To synthesize the OpenCL kernel:  0

	- aoc -v -report -profile -O3 -fp-relaxed kernel_name.cl -o bin/backprojection3D_kernel.aocx -board="board_name"  

Requirements:  

	- Intel FPGA SDK for OpenCL  
	- The device right BSP for the device)  

