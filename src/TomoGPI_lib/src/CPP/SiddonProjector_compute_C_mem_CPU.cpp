/*
 *
 * Authors: chapdelaine, gac
 */

#include "Projector_CPU.cuh"


#define UBLOCK_SIZE 16
#define VBLOCK_SIZE 16 
#define PHIBLOCK_SIZE 8

/* RegularSamplingProjector definition */
template <typename T>
SiddonProjector_compute_C_mem_CPU<T>::SiddonProjector_compute_C_mem_CPU(Acquisition* acquisition, Detector* detector, Volume_CPU<T>* volume) : Projector<Volume_CPU,Sinogram3D_CPU,T>(acquisition, detector, volume){}

template <typename T>
SiddonProjector_compute_C_mem_CPU<T>::~SiddonProjector_compute_C_mem_CPU(){}

template <typename T>
void SiddonProjector_compute_C_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume)
{

	std::cout << "\tSiddon Projection all on CPU" << std::endl;
	this->setVolume(volume);
	// data
	T* d_volume=this->getVolume()->getVolumeData();
	T* sinogram_g=estimatedSinogram->getDataSinogram();

	// constant
	float focusDetectorDistance_GPU = this->getAcquisition()->getFocusDetectorDistance();
	float focusObjectDistance_GPU = this->getAcquisition()->getFocusObjectDistance();
	float xVolumeCenterPixel_GPU = this->getVolume()->getXVolumeCenterPixel();
	float yVolumeCenterPixel_GPU = this->getVolume()->getYVolumeCenterPixel();
	float zVolumeCenterPixel_GPU = this->getVolume()->getZVolumeCenterPixel();
	float xVolumePixelSize_GPU = this->getVolume()->getXVolumePixelSize();
	unsigned long int xVolumePixelNb_GPU = this->getVolume()->getXVolumePixelNb();
	unsigned long int yVolumePixelNb_GPU = this->getVolume()->getYVolumePixelNb();
	unsigned long int zVolumePixelNb_GPU = this->getVolume()->getZVolumePixelNb();
	float uDetectorCenterPixel_GPU = this->getDetector()->getUDetectorCenterPixel();
	float vDetectorCenterPixel_GPU = this->getDetector()->getVDetectorCenterPixel();
	float uDetectorPixelSize_GPU = this->getDetector()->getUDetectorPixelSize();
	float vDetectorPixelSize_GPU = this->getDetector()->getVDetectorPixelSize();
	float *alphaIOcylinderC_GPU=this->getAlphaIOcylinderC();
	float *betaIOcylinderC_GPU=this->getBetaIOcylinderC();

	unsigned long long int phi,un_e, vn_e;
	unsigned long long int adresse_une_vne, adresse_xne_yne_zne;
	float x_s,y_s,z_s,x_det,y_det,z_det,L;
	float lambda_min,lambda_max;
	float lambda,lambdax,lambday,lambdaz,lambda_ksi,chord_length;
	float xmin,ymin,zmin;
	float zmax;
	unsigned long long int xn_e,yn_e,zn_e;
	float sino_ray;
	float A, B, C, delta_lambda;
	float s,t;

	unsigned long int uSinogramPixelNb_GPU=this->getDetector()->getUDetectorPixelNb();
	unsigned long int vSinogramPixelNb_GPU=this->getDetector()->getVDetectorPixelNb();
	unsigned long int projectionSinogramNb=this->getAcquisition()->getProjectionNb();

	//origine
	float x_Lr=xVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
	float y_Lr=yVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
	float z_Lr=zVolumeCenterPixel_GPU*xVolumePixelSize_GPU;

for(uint blockphi = 0; blockphi < projectionSinogramNb; blockphi+=PHIBLOCK_SIZE){
	for(uint blockv = 0; blockv < vSinogramPixelNb_GPU; blockv+=VBLOCK_SIZE){
		for(uint blocku = 0; blocku < uSinogramPixelNb_GPU; blocku+=UBLOCK_SIZE){

			for(phi=blockphi;phi<blockphi + PHIBLOCK_SIZE;phi++){
				//phi  = l_phi + PHIBLOCK_SIZE;

				//Coord de la source
				x_s = x_Lr+focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
				y_s = y_Lr+focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
				z_s= z_Lr;

				for(vn_e=blockv;vn_e<blockv + VBLOCK_SIZE;vn_e++){
					for(un_e=blocku;un_e<blocku + UBLOCK_SIZE;un_e++){

						//un_e = l_un_e + UBLOCK_SIZE;
						//vn_e = l_vn_e + VBLOCK_SIZE;

						// initialize
						sino_ray=0;

						// s et t
						s=((float)un_e - uDetectorCenterPixel_GPU)*uDetectorPixelSize_GPU;
						t=((float)vn_e-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;

						//Coord du detecteur
						x_det = x_Lr-alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU)-s*betaIOcylinderC_GPU[phi];
						y_det = y_Lr-betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU)+s*alphaIOcylinderC_GPU[phi];
						z_det = z_Lr+t;

						//Calcul de la longueur du rayon
						L=sqrtf(focusDetectorDistance_GPU*focusDetectorDistance_GPU+s*s+t*t);

						//calcul de lambda_min et lambda_max
						A=(focusDetectorDistance_GPU*focusDetectorDistance_GPU + s*s)/(L*L);
						B=(-focusObjectDistance_GPU*focusDetectorDistance_GPU)/L;
						C=focusObjectDistance_GPU*focusObjectDistance_GPU-0.25*((float) xVolumePixelNb_GPU)*((float) xVolumePixelNb_GPU)*xVolumePixelSize_GPU*xVolumePixelSize_GPU;
						delta_lambda=B*B-A*C;
						//std::cout<<"delta_lambda : "<<to_string(delta_lambda)<<std::endl;

						if(delta_lambda>0.0){

							lambda_min=(-B-sqrtf(delta_lambda))/A;
							lambda_max=(-B+sqrtf(delta_lambda))/A;


							xmin=(x_s+(lambda_min*(x_det-x_s)/L))/xVolumePixelSize_GPU;
							ymin=(y_s+(lambda_min*(y_det-y_s)/L))/xVolumePixelSize_GPU;
							zmin=(z_s+(lambda_min*(z_det-z_s)/L))/xVolumePixelSize_GPU;
							//printf("%.1f %.1f %.1f\t", xmin, ymin, zmin);

							// first voxels indices and initialize lambdax, lambday, lambdaz
							if(x_s<=x_det){
								xmin=xmin-0.5;
								if(xmin<0){
									xn_e=0;
								}else if(xmin>xVolumePixelNb_GPU-1){
									xn_e=xVolumePixelNb_GPU-1;
								}else{
									xn_e=floor(xmin);
								}
								if(fabs(x_det-x_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambdax=L*(((float) xn_e+0.5)*xVolumePixelSize_GPU-x_s)/(x_det-x_s);
								}else{
									lambdax=L;
								}
							}else{
								xmin=xmin+0.5;
								if(xmin<0){
									xn_e=0;
								}else if(xmin>xVolumePixelNb_GPU-1){
									xn_e=xVolumePixelNb_GPU-1;
								}else{
									xn_e=ceil(xmin);
								}
								if(fabs(x_det-x_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambdax=L*(((float) xn_e-0.5)*xVolumePixelSize_GPU-x_s)/(x_det-x_s);
								}else{
									lambdax=L;
								}
							}
							if(y_s<=y_det){
								ymin=ymin-0.5;
								if(ymin<0){
									yn_e=0;
								}else if(ymin>yVolumePixelNb_GPU-1){
									yn_e=yVolumePixelNb_GPU-1;
								}else{
									yn_e=floor(ymin);
								}
								if(fabs(y_det-y_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambday=L*(((float) yn_e+0.5)*xVolumePixelSize_GPU-y_s)/(y_det-y_s);
								}else{
									lambday=L;
								}
							}else{
								ymin=ymin+0.5;
								if(ymin<0){
									yn_e=0;
								}else if(ymin>yVolumePixelNb_GPU-1){
									yn_e=yVolumePixelNb_GPU-1;
								}else{
									yn_e=ceil(ymin);
								}
								if(fabs(y_det-y_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambday=L*(((float) yn_e-0.5)*xVolumePixelSize_GPU-y_s)/(y_det-y_s);
								}else{
									lambday=L;
								}
							}
							if(z_s<=z_det){
								zmin=zmin-0.5;
								if(zmin<0){
									zn_e=0;
								}else if(zmin>zVolumePixelNb_GPU-1){
									zn_e=zVolumePixelNb_GPU-1;
								}else{
									zn_e=floor(zmin);
								}
								if(fabs(z_det-z_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambdaz=L*(((float) zn_e+0.5)*xVolumePixelSize_GPU-z_s)/(z_det-z_s);
								}else{
									lambdaz=L;
								}
							}else{
								zmin=zmin+0.5;
								if(zmin<0){
									zn_e=0;
								}else if(zmin>zVolumePixelNb_GPU-1){
									zn_e=zVolumePixelNb_GPU-1;
								}else{
									zn_e=ceil(zmin);
								}
								if(fabs(z_det-z_s)>xVolumePixelSize_GPU){// avoid division by zero
									lambdaz=L*(((float) zn_e-0.5)*xVolumePixelSize_GPU-z_s)/(z_det-z_s);
								}else{
									lambdaz=L;
								}
							}

							// initialize lambda
							lambda=lambda_min;

							// ray-tracing
							lambda_ksi=0;
							while(lambda<=lambda_max){

								lambda_ksi=lambdax;
								if(lambday<lambda_ksi){
									lambda_ksi=lambday;
								}
								if(lambdaz<lambda_ksi){
									lambda_ksi=lambdaz;
								}

								// chord length
								chord_length=lambda_ksi-lambda;
							
								//printf("%d %d %d\t", xn_e, yn_e, zn_e);
								// sinogram
								adresse_xne_yne_zne=(xn_e%256 + (yn_e%256) * xVolumePixelNb_GPU + (zn_e%256) * xVolumePixelNb_GPU * yVolumePixelNb_GPU);
								sino_ray+=chord_length*d_volume[adresse_xne_yne_zne];
							
								// update
								lambda=lambda_ksi;
								if(lambdax<=lambda_ksi){
									lambdax=lambdax+(L/fabs(x_det-x_s))*xVolumePixelSize_GPU;
									if(x_s<=x_det){
										xn_e=xn_e+1;
									}else{
										xn_e=xn_e-1;
									}
								}
								if(lambday<=lambda_ksi){
									lambday=lambday+(L/fabs(y_det-y_s))*xVolumePixelSize_GPU;
									if(y_s<=y_det){
										yn_e=yn_e+1;
									}else{
										yn_e=yn_e-1;
									}
								}
								if(lambdaz<=lambda_ksi){
									lambdaz=lambdaz+(L/fabs(z_det-z_s))*xVolumePixelSize_GPU;
									if(z_s<=z_det){
										zn_e=zn_e+1;
									}else{
										zn_e=zn_e-1;
									}
								}

							}

							adresse_une_vne=un_e%256 + (vn_e%256) * uSinogramPixelNb_GPU + phi*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
							sinogram_g[adresse_une_vne]=sino_ray;

						}else{
							adresse_une_vne=un_e%256 + (vn_e%256) * uSinogramPixelNb_GPU + phi*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
							sinogram_g[adresse_une_vne]=0;
						}

					}
				}
			}
		}
	}
}

		
}



// Siddon all on CPU
// template <typename T>
// void SiddonProjector_compute_C_mem_CPU<T>::doProjection(Sinogram3D_CPU<T>* estimatedSinogram,Volume_CPU<T>* volume)
// {

// 	std::cout << "\tSiddon Projection all on CPU" << std::endl;
// 	this->setVolume(volume);
// 	// data
// 	T* d_volume=this->getVolume()->getVolumeData();
// 	T* sinogram_g=estimatedSinogram->getDataSinogram();

// 	// constant
// 	float focusDetectorDistance_GPU = this->getAcquisition()->getFocusDetectorDistance();
// 	float focusObjectDistance_GPU = this->getAcquisition()->getFocusObjectDistance();
// 	float xVolumeCenterPixel_GPU = this->getVolume()->getXVolumeCenterPixel();
// 	float yVolumeCenterPixel_GPU = this->getVolume()->getYVolumeCenterPixel();
// 	float zVolumeCenterPixel_GPU = this->getVolume()->getZVolumeCenterPixel();
// 	float xVolumePixelSize_GPU = this->getVolume()->getXVolumePixelSize();
// 	unsigned long int xVolumePixelNb_GPU = this->getVolume()->getXVolumePixelNb();
// 	unsigned long int yVolumePixelNb_GPU = this->getVolume()->getYVolumePixelNb();
// 	unsigned long int zVolumePixelNb_GPU = this->getVolume()->getZVolumePixelNb();
// 	float uDetectorCenterPixel_GPU = this->getDetector()->getUDetectorCenterPixel();
// 	float vDetectorCenterPixel_GPU = this->getDetector()->getVDetectorCenterPixel();
// 	float uDetectorPixelSize_GPU = this->getDetector()->getUDetectorPixelSize();
// 	float vDetectorPixelSize_GPU = this->getDetector()->getVDetectorPixelSize();
// 	float *alphaIOcylinderC_GPU=this->getAlphaIOcylinderC();
// 	float *betaIOcylinderC_GPU=this->getBetaIOcylinderC();

// 	unsigned long long int phi,un_e, vn_e;
// 	unsigned long long int adresse_une_vne, adresse_xne_yne_zne;
// 	float x_s,y_s,z_s,x_det,y_det,z_det,L;
// 	float lambda_min,lambda_max;
// 	float lambda,lambdax,lambday,lambdaz,lambda_ksi,chord_length;
// 	float xmin,ymin,zmin;
// 	float zmax;
// 	unsigned long long int xn_e,yn_e,zn_e;
// 	float sino_ray;
// 	float A, B, C, delta_lambda;
// 	float s,t;

// 	unsigned long int uSinogramPixelNb_GPU=this->getDetector()->getUDetectorPixelNb();
// 	unsigned long int vSinogramPixelNb_GPU=this->getDetector()->getVDetectorPixelNb();
// 	unsigned long int projectionSinogramNb=this->getAcquisition()->getProjectionNb();

// 	//origine
// 	float x_Lr=xVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
// 	float y_Lr=yVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
// 	float z_Lr=zVolumeCenterPixel_GPU*xVolumePixelSize_GPU;

// 	for(phi=0;phi<projectionSinogramNb;phi++){

// 		//Coord de la source
// 		x_s = x_Lr+focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
// 		y_s = y_Lr+focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
// 		z_s= z_Lr;

// 		for(vn_e=0;vn_e<vSinogramPixelNb_GPU;vn_e++){
// 			for(un_e=0;un_e<uSinogramPixelNb_GPU;un_e++){

// 				// initialize
// 				sino_ray=0;

// 				// s et t
// 				s=((float)un_e - uDetectorCenterPixel_GPU)*uDetectorPixelSize_GPU;
// 				t=((float)vn_e-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;

// 				//Coord du detecteur
// 				x_det = x_Lr-alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU)-s*betaIOcylinderC_GPU[phi];
// 				y_det = y_Lr-betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU)+s*alphaIOcylinderC_GPU[phi];
// 				z_det = z_Lr+t;

// 				//Calcul de la longueur du rayon
// 				L=sqrtf(focusDetectorDistance_GPU*focusDetectorDistance_GPU+s*s+t*t);

// 				//calcul de lambda_min et lambda_max
// 				A=(focusDetectorDistance_GPU*focusDetectorDistance_GPU + s*s)/(L*L);
// 				B=(-focusObjectDistance_GPU*focusDetectorDistance_GPU)/L;
// 				C=focusObjectDistance_GPU*focusObjectDistance_GPU-0.25*((float) xVolumePixelNb_GPU)*((float) xVolumePixelNb_GPU)*xVolumePixelSize_GPU*xVolumePixelSize_GPU;
// 				delta_lambda=B*B-A*C;
// 				//std::cout<<"delta_lambda : "<<to_string(delta_lambda)<<std::endl;

// 				if(delta_lambda>0.0){

// 					lambda_min=(-B-sqrtf(delta_lambda))/A;
// 					lambda_max=(-B+sqrtf(delta_lambda))/A;


// 					xmin=(x_s+(lambda_min*(x_det-x_s)/L))/xVolumePixelSize_GPU;
// 					ymin=(y_s+(lambda_min*(y_det-y_s)/L))/xVolumePixelSize_GPU;
// 					zmin=(z_s+(lambda_min*(z_det-z_s)/L))/xVolumePixelSize_GPU;

// 					// first voxels indices and initialize lambdax, lambday, lambdaz
// 					if(x_s<=x_det){
// 						xmin=xmin-0.5;
// 						if(xmin<0){
// 							xn_e=0;
// 						}else if(xmin>xVolumePixelNb_GPU-1){
// 							xn_e=xVolumePixelNb_GPU-1;
// 						}else{
// 							xn_e=floor(xmin);
// 						}
// 						if(fabs(x_det-x_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambdax=L*(((float) xn_e+0.5)*xVolumePixelSize_GPU-x_s)/(x_det-x_s);
// 						}else{
// 							lambdax=L;
// 						}
// 					}else{
// 						xmin=xmin+0.5;
// 						if(xmin<0){
// 							xn_e=0;
// 						}else if(xmin>xVolumePixelNb_GPU-1){
// 							xn_e=xVolumePixelNb_GPU-1;
// 						}else{
// 							xn_e=ceil(xmin);
// 						}
// 						if(fabs(x_det-x_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambdax=L*(((float) xn_e-0.5)*xVolumePixelSize_GPU-x_s)/(x_det-x_s);
// 						}else{
// 							lambdax=L;
// 						}
// 					}
// 					if(y_s<=y_det){
// 						ymin=ymin-0.5;
// 						if(ymin<0){
// 							yn_e=0;
// 						}else if(ymin>yVolumePixelNb_GPU-1){
// 							yn_e=yVolumePixelNb_GPU-1;
// 						}else{
// 							yn_e=floor(ymin);
// 						}
// 						if(fabs(y_det-y_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambday=L*(((float) yn_e+0.5)*xVolumePixelSize_GPU-y_s)/(y_det-y_s);
// 						}else{
// 							lambday=L;
// 						}
// 					}else{
// 						ymin=ymin+0.5;
// 						if(ymin<0){
// 							yn_e=0;
// 						}else if(ymin>yVolumePixelNb_GPU-1){
// 							yn_e=yVolumePixelNb_GPU-1;
// 						}else{
// 							yn_e=ceil(ymin);
// 						}
// 						if(fabs(y_det-y_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambday=L*(((float) yn_e-0.5)*xVolumePixelSize_GPU-y_s)/(y_det-y_s);
// 						}else{
// 							lambday=L;
// 						}
// 					}
// 					if(z_s<=z_det){
// 						zmin=zmin-0.5;
// 						if(zmin<0){
// 							zn_e=0;
// 						}else if(zmin>zVolumePixelNb_GPU-1){
// 							zn_e=zVolumePixelNb_GPU-1;
// 						}else{
// 							zn_e=floor(zmin);
// 						}
// 						if(fabs(z_det-z_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambdaz=L*(((float) zn_e+0.5)*xVolumePixelSize_GPU-z_s)/(z_det-z_s);
// 						}else{
// 							lambdaz=L;
// 						}
// 					}else{
// 						zmin=zmin+0.5;
// 						if(zmin<0){
// 							zn_e=0;
// 						}else if(zmin>zVolumePixelNb_GPU-1){
// 							zn_e=zVolumePixelNb_GPU-1;
// 						}else{
// 							zn_e=ceil(zmin);
// 						}
// 						if(fabs(z_det-z_s)>xVolumePixelSize_GPU){// avoid division by zero
// 							lambdaz=L*(((float) zn_e-0.5)*xVolumePixelSize_GPU-z_s)/(z_det-z_s);
// 						}else{
// 							lambdaz=L;
// 						}
// 					}

// 					// initialize lambda
// 					lambda=lambda_min;

// 					// ray-tracing
// 					lambda_ksi=0;
// 					while(lambda<=lambda_max){

// 						lambda_ksi=lambdax;
// 						if(lambday<lambda_ksi){
// 							lambda_ksi=lambday;
// 						}
// 						if(lambdaz<lambda_ksi){
// 							lambda_ksi=lambdaz;
// 						}

// 						// chord length
// 						chord_length=lambda_ksi-lambda;
					
// 						printf("%d %d %d\t", xn_e, yn_e, zn_e);
// 						// sinogram
// 						adresse_xne_yne_zne=(xn_e%256 + (yn_e%256) * xVolumePixelNb_GPU + (zn_e%256) * xVolumePixelNb_GPU * yVolumePixelNb_GPU);
// 						sino_ray+=chord_length*d_volume[adresse_xne_yne_zne];
					
// 						// update
// 						lambda=lambda_ksi;
// 						if(lambdax<=lambda_ksi){
// 							lambdax=lambdax+(L/fabs(x_det-x_s))*xVolumePixelSize_GPU;
// 							if(x_s<=x_det){
// 								xn_e=xn_e+1;
// 							}else{
// 								xn_e=xn_e-1;
// 							}
// 						}
// 						if(lambday<=lambda_ksi){
// 							lambday=lambday+(L/fabs(y_det-y_s))*xVolumePixelSize_GPU;
// 							if(y_s<=y_det){
// 								yn_e=yn_e+1;
// 							}else{
// 								yn_e=yn_e-1;
// 							}
// 						}
// 						if(lambdaz<=lambda_ksi){
// 							lambdaz=lambdaz+(L/fabs(z_det-z_s))*xVolumePixelSize_GPU;
// 							if(z_s<=z_det){
// 								zn_e=zn_e+1;
// 							}else{
// 								zn_e=zn_e-1;
// 							}
// 						}

// 					}

// 					adresse_une_vne=un_e%256 + (vn_e%256) * uSinogramPixelNb_GPU + phi*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
// 					sinogram_g[adresse_une_vne]=sino_ray;

// 				}else{
// 					adresse_une_vne=un_e%256 + (vn_e%256) * uSinogramPixelNb_GPU + phi*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
// 					sinogram_g[adresse_une_vne]=0;
// 				}

// 			}
// 		}
// 	}
// }


template <typename T>
void SiddonProjector_compute_C_mem_CPU<T>::EnableP2P(){}

template <typename T>
void SiddonProjector_compute_C_mem_CPU<T>::DisableP2P(){}

#include "Projector_instances_CPU.cu"