 /*
 * projection3D_Siddon.cl
 *
 *      Author: diakite
 */

//#include "opencl_compat.h"

#define xVolumePixelNb_GPU 256
#define yVolumePixelNb_GPU 256
#define zVolumePixelNb_GPU 256

#define uSinogramPixelNb_GPU 256
#define vSinogramPixelNb_GPU 256
#define projectionSinogramNb 256

#define xVolumePixelNb_GPU_inverse 1/256
#define yVolumePixelNb_GPU_inverse 1/256
#define zVolumePixelNb_GPU_inverse 1/256
#define DIMPHI 256

//__attribute__((num_simd_work_items(4)))
//__attribute__((num_compute_units(8)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void projection3D(__global const float * restrict volume,
				__global float * restrict sinogram,
				__global float * restrict alpha,
				__global float * restrict beta,
				float8 host_constant,
				float2 host_delta)
{

		// constant
		float focusDetectorDistance_GPU = host_constant.s0;
		float focusObjectDistance_GPU = host_constant.s1;
		float xVolumeCenterPixel_GPU = host_constant.s2;
		float yVolumeCenterPixel_GPU = host_constant.s3;
		float zVolumeCenterPixel_GPU = host_constant.s4;
		float xVolumePixelSize_GPU = host_constant.s5;
		float uDetectorCenterPixel_GPU = host_constant.s6;
		float vDetectorCenterPixel_GPU = host_constant.s7;
		float uDetectorPixelSize_GPU = host_delta.s0;
		float vDetectorPixelSize_GPU = host_delta.s1;
		//float *alphaIOcylinderC_GPU=constante_GPU->alpha_wn;//sinus -- this->alphaIOcylinderC;
		//float *betaIOcylinderC_GPU=constante_GPU->beta_wn;//cosinus -- this->betaIOcylinderC;
		//printf("%d\t", host_delta.s0);
		unsigned int phi,un_e, vn_e;
		unsigned int adresse_une_vne, adresse_xne_yne_zne;
		float x_s,y_s,z_s,x_det,y_det,z_det,L,L_inverse;
		float lambda_min,lambda_max;
		float lambda,lambdax,lambday,lambdaz,lambda_ksi,chord_length;
		float xmin,ymin,zmin;
		float zmax;
		unsigned int xn_e,yn_e,zn_e;
		float sino_ray;
		float A,A_inverse, B, C, delta_lambda,local_alpha, local_beta;
		float s,t;

		/*unsigned int uSinogramPixelNb_GPU=sampling->N_un;
		unsigned int vSinogramPixelNb_GPU=sampling->N_vn;
		unsigned int projectionSinogramNb=sampling->N_phi;*/

		//origine
		float x_Lr=xVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
		float y_Lr=yVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
		float z_Lr=zVolumeCenterPixel_GPU*xVolumePixelSize_GPU;
	
		un_e = get_global_id(0);
		vn_e = get_global_id(1);
		phi = get_global_id(2);
	
		
			local_alpha=alpha[phi];
			local_beta=beta[phi];
			//Coord de la source
			x_s = x_Lr+focusObjectDistance_GPU*local_alpha;
			y_s = y_Lr+focusObjectDistance_GPU*local_beta;
			z_s= z_Lr;

					// initialize
					sino_ray=0.0f;

					// s et t
					s=((float)un_e - uDetectorCenterPixel_GPU)*uDetectorPixelSize_GPU;
					t=((float)vn_e-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;

					//Coord du detecteur
					x_det = x_Lr-local_alpha*(focusDetectorDistance_GPU-focusObjectDistance_GPU)-s*local_beta;
					y_det = y_Lr-local_beta*(focusDetectorDistance_GPU-focusObjectDistance_GPU)+s*local_alpha;
					z_det = z_Lr+t;

					//Calcul de la longueur du rayon
					L=sqrt(focusDetectorDistance_GPU * focusDetectorDistance_GPU + s * s + t * t);
					L_inverse=1.0f/L;

					//calcul de lambda_min et lambda_max
					A=(focusDetectorDistance_GPU * focusDetectorDistance_GPU + s * s)*(L_inverse * L_inverse);
					A_inverse = 1.0f/A;
					B = (-focusObjectDistance_GPU * focusDetectorDistance_GPU) * L_inverse;
					C = focusObjectDistance_GPU * focusObjectDistance_GPU - 0.25f * ((float) xVolumePixelNb_GPU)*((float) xVolumePixelNb_GPU)* xVolumePixelSize_GPU * xVolumePixelSize_GPU;
					delta_lambda = B * B - A * C;
					//std::cout<<"delta_lambda : "<<to_string(delta_lambda)<<std::endl;
					if(delta_lambda > 0.0f){

						lambda_min = (- B - sqrt(fabs(delta_lambda))) * A_inverse;
						lambda_max = (- B + sqrt(fabs(delta_lambda))) * A_inverse;


						xmin = (x_s + (lambda_min * (x_det - x_s) / L)) / xVolumePixelSize_GPU;
						ymin = (y_s + (lambda_min * (y_det - y_s) / L)) / xVolumePixelSize_GPU;
						zmin = (z_s + (lambda_min * (z_det - z_s) / L)) / xVolumePixelSize_GPU;


						// first voxels indices and initialize lambdax, lambday, lambdaz
						if(x_s <= x_det){
							xmin = xmin - 0.5f;
							if(xmin < 0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e = xVolumePixelNb_GPU - 1;
							}else{
								xn_e = floor(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax = L * (((float) xn_e + 0.5f) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}else{
							xmin = xmin + 0.5f;
							if(xmin < 0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e = xVolumePixelNb_GPU - 1;
							}else{
								xn_e = ceil(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax = L * (((float) xn_e - 0.5f) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}
						if(y_s <= y_det){
							ymin = ymin - 0.5f;
							if(ymin<0){
								yn_e=0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = floor(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e + 0.5f) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}else{
							ymin = ymin + 0.5f;
							if(ymin < 0){
								yn_e = 0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = ceil(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e - 0.5f) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}
						if(z_s <= z_det){
							zmin = zmin - 0.5f;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = floor(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e + 0.5f) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}else{
							zmin = zmin + 0.5f;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = ceil(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e - 0.5f) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}

						// initialize lambda
						lambda=lambda_min;

						// ray-tracing
						lambda_ksi=0;
					
					
						while(lambda<=lambda_max){
						/*
						for(int i=0; i<32; ++i){*/

							lambda_ksi = lambdax;
							if(lambday < lambda_ksi){
								lambda_ksi = lambday;
							}
							if(lambdaz < lambda_ksi){
								lambda_ksi = lambdaz;
							}

							// chord length
							chord_length = lambda_ksi - lambda;

							// sinogram
							adresse_xne_yne_zne = xn_e + yn_e * xVolumePixelNb_GPU + zn_e*xVolumePixelNb_GPU * yVolumePixelNb_GPU;
							sino_ray += chord_length * volume[adresse_xne_yne_zne];

							// update
							lambda = lambda_ksi;
							if(lambdax <= lambda_ksi){
								lambdax = lambdax + (L / fabs(x_det - x_s)) * xVolumePixelSize_GPU;
								if(x_s <= x_det){
									xn_e = (xn_e + 1)%256;
								}else{
									xn_e = (xn_e - 1)%256;
								}
							}
							if(lambday <= lambda_ksi){
								lambday = lambday + (L / fabs(y_det - y_s)) * xVolumePixelSize_GPU;
								if(y_s <= y_det){
									yn_e = (yn_e + 1)%256;
								}else{
									yn_e = (yn_e - 1)%256;
								}
							}
							if(lambdaz <= lambda_ksi){
								lambdaz = lambdaz + (L / fabs(z_det - z_s)) * xVolumePixelSize_GPU;
								if(z_s <= z_det){
									zn_e = (zn_e + 1)%256;
								}else{
									zn_e = (zn_e - 1)%256;
								} 
							}

						}
						
					


					
					}else{
						sino_ray=0.0f;
					}

					adresse_une_vne = un_e + vn_e * uSinogramPixelNb_GPU + phi * uSinogramPixelNb_GPU * vSinogramPixelNb_GPU;
					sinogram[adresse_une_vne] = sino_ray;
	}



__kernel void projection3D_SWI(__global const float * restrict volume,
				 __global float * restrict sinogram,
				 __global float * restrict alpha,
				 __global float * restrict beta,
				 float8 host_constant,
				 float2 host_delta)
{

		// constant
		float focusDetectorDistance_GPU = host_constant.s0;
		float focusObjectDistance_GPU = host_constant.s1;
		float xVolumeCenterPixel_GPU = host_constant.s2;
		float yVolumeCenterPixel_GPU = host_constant.s3;
		float zVolumeCenterPixel_GPU = host_constant.s4;
		float xVolumePixelSize_GPU = host_constant.s5;
		float uDetectorCenterPixel_GPU = host_constant.s6;
		float vDetectorCenterPixel_GPU = host_constant.s7;
		float uDetectorPixelSize_GPU = host_delta.s0;
		float vDetectorPixelSize_GPU = host_delta.s1;
		unsigned int phi,un_e, vn_e;
		unsigned int adresse_une_vne, adresse_xne_yne_zne;
		float x_s,y_s,z_s,x_det,y_det,z_det,L;
		float lambda_min,lambda_max;
		float lambda,lambdax,lambday,lambdaz,lambda_ksi,chord_length;
		float xmin,ymin,zmin;
		float zmax;
		unsigned int xn_e,yn_e,zn_e;
		float sino_ray;
		float A,A_inverse, B, C, delta_lambda,local_alpha, local_beta;
		float s,t;


		//origine
		float x_Lr = xVolumeCenterPixel_GPU * xVolumePixelSize_GPU;
		float y_Lr = yVolumeCenterPixel_GPU * xVolumePixelSize_GPU;
		float z_Lr = zVolumeCenterPixel_GPU * xVolumePixelSize_GPU;

	
		for(phi = 0; phi < projectionSinogramNb; phi++){
		
			local_alpha = alpha[phi];
			local_beta = beta[phi];
			//Coord de la source
			x_s = x_Lr + focusObjectDistance_GPU * local_alpha;
			y_s = y_Lr + focusObjectDistance_GPU * local_beta;
			z_s= z_Lr;

			for(vn_e = 0; vn_e < vSinogramPixelNb_GPU; vn_e++){
				for(un_e = 0; un_e < uSinogramPixelNb_GPU; un_e++){

					// initialize
					sino_ray = 0.0f;

					// s et t
					s = ((float) un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU;
					t = ((float) vn_e - vDetectorCenterPixel_GPU) * vDetectorPixelSize_GPU;

				//Coord du detecteur
					x_det = x_Lr - local_alpha * (focusDetectorDistance_GPU - focusObjectDistance_GPU) - s * local_beta;
					y_det = y_Lr - local_beta * (focusDetectorDistance_GPU - focusObjectDistance_GPU) + s * local_alpha;
					z_det = z_Lr + t;

					//Calcul de la longueur du rayon
					L = sqrt(fabs((focusDetectorDistance_GPU*focusDetectorDistance_GPU) + (s*s) + (t*t)));
				//	printf("s=%f t=%f L=%f\t", s, t, L);printf("\n");

					//calcul de lambda_min et lambda_max
					A = (focusDetectorDistance_GPU * focusDetectorDistance_GPU + s*s)/(L*L);
					// B = (-focusObjectDistance_GPU * focusDetectorDistance_GPU)/L;
					 B = (-98.0f * 230.0f)/L;
					C = focusObjectDistance_GPU * focusObjectDistance_GPU - 0.25f *((float) xVolumePixelNb_GPU)*((float) xVolumePixelNb_GPU) * xVolumePixelSize_GPU * xVolumePixelSize_GPU;
					delta_lambda = B * B - A * C;
					// printf("A=%f B=%f C=%f\t", A,B,C);
					if(delta_lambda > 0.0){

						lambda_min = (- B - sqrt(fabs(delta_lambda))) / A;
						lambda_max = (- B + sqrt(fabs(delta_lambda))) / A;
						// printf("lamd=%f\t", lambda_max);

						xmin = (x_s + (lambda_min * (x_det - x_s) / L)) / xVolumePixelSize_GPU;
						ymin = (y_s + (lambda_min * (y_det - y_s) / L)) / xVolumePixelSize_GPU;
						zmin = (z_s + (lambda_min * (z_det - z_s) / L)) / xVolumePixelSize_GPU;

						// first voxels indices and initialize lambdax, lambday, lambdaz
						if(x_s <= x_det){
							xmin = xmin - 0.5;
							if(xmin < 0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e= xVolumePixelNb_GPU - 1;
							}else{
								xn_e = floor(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax= L * (((float) xn_e + 0.5) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}else{
							xmin = xmin + 0.5;
							if(xmin<0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e = xVolumePixelNb_GPU - 1;
							}else{
								xn_e = ceil(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax = L * (((float) xn_e - 0.5) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}
						if(y_s <= y_det){
							ymin = ymin - 0.5;
							if(ymin < 0){
								yn_e = 0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = floor(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e + 0.5) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}else{
							ymin= ymin + 0.5;
							if(ymin < 0){
								yn_e = 0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = ceil(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e - 0.5) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}
						if(z_s <= z_det){
							zmin = zmin - 0.5;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = floor(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e + 0.5) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}else{
							zmin = zmin + 0.5;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = ceil(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e - 0.5) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}

						// initialize lambda
						lambda = lambda_min;

						// ray-tracing
						lambda_ksi = 0;
		 // for(unsigned int i = 0; i <= 64; ++i){
				while(lambda <= lambda_max){

							lambda_ksi = lambdax;
							if(lambday < lambda_ksi){
								lambda_ksi = lambday;
							}
							if(lambdaz < lambda_ksi){
								lambda_ksi = lambdaz;
							}

							// chord length
							chord_length = lambda_ksi - lambda;
					// 		printf("Kernel execution");

							// sinogram
							adresse_xne_yne_zne = (xn_e + yn_e * xVolumePixelNb_GPU + zn_e * xVolumePixelNb_GPU * yVolumePixelNb_GPU);
							sino_ray += chord_length * volume[adresse_xne_yne_zne];


							// update
							lambda = lambda_ksi;
							if(lambdax <= lambda_ksi){
								lambdax = lambdax + (L / fabs(x_det - x_s)) * xVolumePixelSize_GPU;
								if(x_s <= x_det){
									xn_e = (xn_e + 1)%256;
								}else{
									xn_e = (xn_e - 1)%256;
								}
							}
							if(lambday <= lambda_ksi){
								lambday = lambday + (L / fabs(y_det - y_s)) * xVolumePixelSize_GPU;
								if(y_s <= y_det){
									yn_e = (yn_e + 1)%256;
								}else{
									yn_e = (yn_e - 1)%256;
								}
							}
							if(lambdaz <= lambda_ksi){
								lambdaz = lambdaz + (L / fabs(z_det - z_s)) * xVolumePixelSize_GPU;
								if(z_s <= z_det){
									zn_e = (zn_e + 1)%256;
								}else{
									zn_e = (zn_e - 1)%256;
								} 
							}

						}

						adresse_une_vne = un_e + vn_e * uSinogramPixelNb_GPU + phi * uSinogramPixelNb_GPU * vSinogramPixelNb_GPU;
						sinogram[adresse_une_vne] = sino_ray;
 
				}else{
						adresse_une_vne = un_e + vn_e * uSinogramPixelNb_GPU + phi * uSinogramPixelNb_GPU * vSinogramPixelNb_GPU;
						sinogram[adresse_une_vne]=0;
					}
				}
			}
		}
}


__kernel void projection3D_reuse(__global const float * restrict volume,
				 __global float * restrict sinogram,
				 __global float * restrict alpha,
				 __global float * restrict beta,
				 float8 host_constant,
				 float2 host_delta)
{

		// constant
		float focusDetectorDistance_GPU = host_constant.s0;
		float focusObjectDistance_GPU = host_constant.s1;
		float xVolumeCenterPixel_GPU = host_constant.s2;
		float yVolumeCenterPixel_GPU = host_constant.s3;
		float zVolumeCenterPixel_GPU = host_constant.s4;
		float xVolumePixelSize_GPU = host_constant.s5;
		float uDetectorCenterPixel_GPU = host_constant.s6;
		float vDetectorCenterPixel_GPU = host_constant.s7;
		float uDetectorPixelSize_GPU = host_delta.s0;
		float vDetectorPixelSize_GPU = host_delta.s1;
		unsigned int phi,un_e, vn_e;
		unsigned int adresse_une_vne, adresse_xne_yne_zne;
		float x_s,y_s,z_s,x_det,y_det,z_det,L;
		float lambda_min,lambda_max;
		float lambda,lambdax,lambday,lambdaz,lambda_ksi,chord_length;
		float xmin,ymin,zmin;
		float zmax;
		unsigned int xn_e,yn_e,zn_e;
		float sino_ray;
		float A,A_inverse, B, C, delta_lambda,local_alpha, local_beta;
		float s,t;


		//origine
		float x_Lr = xVolumeCenterPixel_GPU * xVolumePixelSize_GPU;
		float y_Lr = yVolumeCenterPixel_GPU * xVolumePixelSize_GPU;
		float z_Lr = zVolumeCenterPixel_GPU * xVolumePixelSize_GPU;

	
		for(phi = 0; phi < projectionSinogramNb; phi++){
		
			local_alpha = alpha[phi];
			local_beta = beta[phi];
			//Coord de la source
			x_s = x_Lr + focusObjectDistance_GPU * local_alpha;
			y_s = y_Lr + focusObjectDistance_GPU * local_beta;
			z_s= z_Lr;

			for(vn_e = 0; vn_e < vSinogramPixelNb_GPU; vn_e++){
				for(un_e = 0; un_e < uSinogramPixelNb_GPU; un_e++){

					// initialize
					sino_ray = 0.0f;

					// s et t
					s = ((float) un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU;
					t = ((float) vn_e - vDetectorCenterPixel_GPU) * vDetectorPixelSize_GPU;

				//Coord du detecteur
					x_det = x_Lr - local_alpha * (focusDetectorDistance_GPU - focusObjectDistance_GPU) - s * local_beta;
					y_det = y_Lr - local_beta * (focusDetectorDistance_GPU - focusObjectDistance_GPU) + s * local_alpha;
					z_det = z_Lr + t;

					//Calcul de la longueur du rayon
					L = sqrt(fabs((focusDetectorDistance_GPU*focusDetectorDistance_GPU) + (s*s) + (t*t)));
				//	printf("s=%f t=%f L=%f\t", s, t, L);printf("\n");

					//calcul de lambda_min et lambda_max
					A = (focusDetectorDistance_GPU * focusDetectorDistance_GPU + s*s)/(L*L);
					// B = (-focusObjectDistance_GPU * focusDetectorDistance_GPU)/L;
					 B = (-98.0f * 230.0f)/L;
					C = focusObjectDistance_GPU * focusObjectDistance_GPU - 0.25f *((float) xVolumePixelNb_GPU)*((float) xVolumePixelNb_GPU) * xVolumePixelSize_GPU * xVolumePixelSize_GPU;
					delta_lambda = B * B - A * C;
					// printf("A=%f B=%f C=%f\t", A,B,C);
					if(delta_lambda > 0.0){

						lambda_min = (- B - sqrt(fabs(delta_lambda))) / A;
						lambda_max = (- B + sqrt(fabs(delta_lambda))) / A;
						// printf("lamd=%f\t", lambda_max);

						xmin = (x_s + (lambda_min * (x_det - x_s) / L)) / xVolumePixelSize_GPU;
						ymin = (y_s + (lambda_min * (y_det - y_s) / L)) / xVolumePixelSize_GPU;
						zmin = (z_s + (lambda_min * (z_det - z_s) / L)) / xVolumePixelSize_GPU;

						// first voxels indices and initialize lambdax, lambday, lambdaz
						if(x_s <= x_det){
							xmin = xmin - 0.5;
							if(xmin < 0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e= xVolumePixelNb_GPU - 1;
							}else{
								xn_e = floor(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax= L * (((float) xn_e + 0.5) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}else{
							xmin = xmin + 0.5;
							if(xmin<0){
								xn_e = 0;
							}else if(xmin > xVolumePixelNb_GPU - 1){
								xn_e = xVolumePixelNb_GPU - 1;
							}else{
								xn_e = ceil(xmin);
							}
							if(fabs(x_det - x_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdax = L * (((float) xn_e - 0.5) * xVolumePixelSize_GPU - x_s)/(x_det - x_s);
							}else{
								lambdax = L;
							}
						}
						if(y_s <= y_det){
							ymin = ymin - 0.5;
							if(ymin < 0){
								yn_e = 0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = floor(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e + 0.5) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}else{
							ymin= ymin + 0.5;
							if(ymin < 0){
								yn_e = 0;
							}else if(ymin > yVolumePixelNb_GPU - 1){
								yn_e = yVolumePixelNb_GPU - 1;
							}else{
								yn_e = ceil(ymin);
							}
							if(fabs(y_det - y_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambday = L * (((float) yn_e - 0.5) * xVolumePixelSize_GPU - y_s)/(y_det - y_s);
							}else{
								lambday = L;
							}
						}
						if(z_s <= z_det){
							zmin = zmin - 0.5;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = floor(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e + 0.5) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}else{
							zmin = zmin + 0.5;
							if(zmin < 0){
								zn_e = 0;
							}else if(zmin > zVolumePixelNb_GPU - 1){
								zn_e = zVolumePixelNb_GPU - 1;
							}else{
								zn_e = ceil(zmin);
							}
							if(fabs(z_det - z_s) > xVolumePixelSize_GPU){// avoid division by zero
								lambdaz = L * (((float) zn_e - 0.5) * xVolumePixelSize_GPU - z_s)/(z_det - z_s);
							}else{
								lambdaz = L;
							}
						}

						// initialize lambda
						lambda = lambda_min;

						// ray-tracing
						lambda_ksi = 0;
		 // for(unsigned int i = 0; i <= 64; ++i){
				while(lambda <= lambda_max){

							lambda_ksi = lambdax;
							if(lambday < lambda_ksi){
								lambda_ksi = lambday;
							}
							if(lambdaz < lambda_ksi){
								lambda_ksi = lambdaz;
							}

							// chord length
							chord_length = lambda_ksi - lambda;
					// 		printf("Kernel execution");

							// sinogram
							adresse_xne_yne_zne = (xn_e + yn_e * xVolumePixelNb_GPU + zn_e * xVolumePixelNb_GPU * yVolumePixelNb_GPU);
							sino_ray += chord_length * volume[adresse_xne_yne_zne];
				

							// update
							lambda = lambda_ksi;
							if(lambdax <= lambda_ksi){
								lambdax = lambdax + (L / fabs(x_det - x_s)) * xVolumePixelSize_GPU;
								if(x_s <= x_det){
									xn_e = (xn_e + 1)%256;
								}else{
									xn_e = (xn_e - 1)%256;
								}
							}
							if(lambday <= lambda_ksi){
								lambday = lambday + (L / fabs(y_det - y_s)) * xVolumePixelSize_GPU;
								if(y_s <= y_det){
									yn_e = (yn_e + 1)%256;
								}else{
									yn_e = (yn_e - 1)%256;
								}
							}
							if(lambdaz <= lambda_ksi){
								lambdaz = lambdaz + (L / fabs(z_det - z_s)) * xVolumePixelSize_GPU;
								if(z_s <= z_det){
									zn_e = (zn_e + 1)%256;
								}else{
									zn_e = (zn_e - 1)%256;
								} 
							}

						}

						adresse_une_vne = un_e + vn_e * uSinogramPixelNb_GPU + phi * uSinogramPixelNb_GPU * vSinogramPixelNb_GPU;
						sinogram[adresse_une_vne] = sino_ray;
 
				}else{
						adresse_une_vne = un_e + vn_e * uSinogramPixelNb_GPU + phi * uSinogramPixelNb_GPU * vSinogramPixelNb_GPU;
						sinogram[adresse_une_vne]=0;
					}
				}
			}
		}
}