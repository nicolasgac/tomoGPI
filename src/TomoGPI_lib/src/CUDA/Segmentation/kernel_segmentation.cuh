
#ifndef _KERNEL_SEGMENTATION_H_
#define _KERNEL_SEGMENTATION_H_



template <typename T>
__global__ void kernel_maxLabelsMGINoirs(T* data_volume, int* labels)
{
	//dimension
	unsigned long int xn = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned long int yn = threadIdx.y+blockIdx.y*blockDim.y;
	unsigned long int zn = threadIdx.z+blockIdx.z*blockDim.z;

	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
	//labels[xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU)] = xn%5;

	int k,k_max,k_neigh=0;
	double proba_max;
	double likelihood = 0;

	bool in_blancs;

	in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
	in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
	in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
	in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));

	unsigned long long int adresse_voxel = xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU);

	unsigned int* count_neighbours=(unsigned int*) malloc(Kclasse*sizeof(unsigned int));
	double *proba_class = (double*) malloc( Kclasse*sizeof(int));

	//unsigned long int xyn = xn + yn;
	// count neighbours in each class

	if( !in_blancs  ){
		//unsigned long long int adresse_voxel = xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU);
		adresse_voxel = 0;
		in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
		in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
		in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
		in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
		if(in_blancs){
			// count neighbours in each class
			for(k=0;k<Kclasse;k++){
				count_neighbours[k]=0;
				proba_class[k] = 0.0;
			}
			//(-1,0,0)
			if(xn>0){
				//k_neigh=labels[(xn-1)+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(1,0,0)
			if(xn<xVolumePixelNb_GPU-1){
				//k_neigh=labels[(xn+1)+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,-1,0)
			if(yn>0){
				//k_neigh=labels[xn+(yn-1)*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,1,0)
			if(yn<yVolumePixelNb_GPU-1){
				//k_neigh=labels[xn+(yn+1)*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,0,-1)
			if(zn>0){
				//k_neigh=labels[xn+yn*xVolumePixelNb_GPU+(zn-1)*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,0,1)
			if(zn<zVolumePixelNb_GPU-1){
				//k_neigh=labels[xn+yn*xVolumePixelNb_GPU+(zn+1)*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//maximum of probability
			likelihood=energySingleton[0]-0.5*log(varianceclasses[0])-0.5*(((double(data_volume[adresse_voxel])-meanclasses[0])*(double(data_volume[adresse_voxel])-meanclasses[0]))/varianceclasses[0]);
			proba_class[0]=likelihood+gammaPotts*(double(count_neighbours[0]));
			k_max=0;
			proba_max=proba_class[0];

			for(k=1;k<Kclasse;k++){
				likelihood=energySingleton[k]-0.5*log(varianceclasses[k])-0.5*(((double(data_volume[adresse_voxel])-meanclasses[k])*(double(data_volume[adresse_voxel])-meanclasses[k]))/varianceclasses[k]);
				proba_class[k]=likelihood+gammaPotts*(double(count_neighbours[k]));
				if(proba_max<proba_class[k]){
					k_max=k;
					proba_max=proba_class[k];
				}
			}
			//label
			//labels[adresse_voxel]=k_max;
		}
		free(count_neighbours);
		free(proba_class);
	}
}


//same code
template <typename T>
__global__ void kernel_maxLabelsMGIBlancs(T* data_volume, int* labels)
{

	unsigned long int xn = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned long int yn = threadIdx.y+blockIdx.y*blockDim.y;
	unsigned long int zn = threadIdx.z+blockIdx.z*blockDim.z;

	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
	//labels[xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU)] = xn%5;

	int k,k_max,k_neigh;
	double proba_max;
	double likelihood = 0;

	bool in_blancs;

	in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
	in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
	in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
	in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));

	unsigned long long int adresse_voxel = xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU);

	unsigned int* count_neighbours=(unsigned int*) malloc(Kclasse*sizeof(unsigned int));
	double *proba_class = (double*) malloc( Kclasse*sizeof(int));

	//unsigned long int xyn = xn + yn;
	// count neighbours in each class

	if( in_blancs ){
		//unsigned long long int adresse_voxel = xn+xVolumePixelNb_GPU*(yn+zn*yVolumePixelNb_GPU);
		adresse_voxel = 0;
		in_blancs=(xn%2==0)&&(yn%2==0)&&(zn%2==0);
		in_blancs=in_blancs||((xn%2==1)&&(yn%2==1)&&(zn%2==0));
		in_blancs=in_blancs||((xn%2==1)&&(yn%2==0)&&(zn%2==1));
		in_blancs=in_blancs||((xn%2==0)&&(yn%2==1)&&(zn%2==1));
		if(in_blancs){
			// count neighbours in each class
			for(k=0;k<Kclasse;k++){
				count_neighbours[k]=0;
				proba_class[k] = 0.0;
			}
			//(-1,0,0)
			if(xn>0){
				//k_neigh=labels[(xn-1)+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(1,0,0)
			if(xn<xVolumePixelNb_GPU-1){
				//k_neigh=labels[(xn+1)+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,-1,0)
			if(yn>0){
				//k_neigh=labels[xn+(yn-1)*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,1,0)
			if(yn<yVolumePixelNb_GPU-1){
				//k_neigh=labels[xn+(yn+1)*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,0,-1)
			if(zn>0){
				//k_neigh=labels[xn+yn*xVolumePixelNb_GPU+(zn-1)*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//(0,0,1)
			if(zn<zVolumePixelNb_GPU-1){
				//k_neigh=labels[xn+yn*xVolumePixelNb_GPU+(zn+1)*xVolumePixelNb_GPU*yVolumePixelNb_GPU];
				count_neighbours[k_neigh]+=1;
			}
			//maximum of probability
			likelihood=energySingleton[0]-0.5*log(varianceclasses[0])-0.5*(((double(data_volume[adresse_voxel])-meanclasses[0])*(double(data_volume[adresse_voxel])-meanclasses[0]))/varianceclasses[0]);
			proba_class[0]=likelihood+gammaPotts*(double(count_neighbours[0]));
			k_max=0;
			proba_max=proba_class[0];

			for(k=1;k<Kclasse;k++){
				likelihood=energySingleton[k]-0.5*log(varianceclasses[k])-0.5*(((double(data_volume[adresse_voxel])-meanclasses[k])*(double(data_volume[adresse_voxel])-meanclasses[k]))/varianceclasses[k]);
				proba_class[k]=likelihood+gammaPotts*(double(count_neighbours[k]));
				if(proba_max<proba_class[k]){
					k_max=k;
					proba_max=proba_class[k];
				}
			}
			//label
			//labels[adresse_voxel]=k_max;
		}
		free(count_neighbours);
		free(proba_class);
	}
}


#endif
