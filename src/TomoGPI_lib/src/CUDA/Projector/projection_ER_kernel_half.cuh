
#ifndef _PROJECTION_ER_KERNEL_HALF_H_
#define _PROJECTION_ER_KERNEL_HALF_H_

#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
//__global__ void
//projection_ERB_kernel_v0(float *sinogram_g,int phi_start, int N_un_start, int N_vn_start, int N_xn_start, int N_yn_start, int N_zn_start,int N_un_sino_temp,int N_vn_sino_temp)

/*! \fn __global__ void projection_ERB_kernel_v0(float *sinogram_g,int phi_start, int N_un_start, int N_vn_start, int N_zn_start) 
  \brief Kernel du projecteur d'echantillonnage regulier pour un objet 3D sur multiGPU
  \param *sinogram_g Sinogramme de sortie (GPU)
  \param phi_start Angle d'initialisation pour le kernel
  \param N_un_start un d'initialisation pour definir le morceau du plan detecteur a calculer
  \param N_vn_start vn d'initialisation pour definir le morceau du plan detecteur a calculer
  \param N_zn_start zn d'initialisation pour definir si on travaille dans le plan nord ou sud du volume
  \param N_un_sino_temp 
  \param N_vn_sino_temp 
  \author Asier Rabanal
 */

__global__ void projection_ERB_kernel_v1_half(unsigned short *sinogram_g,int vn_start, int zn_start)
{
	unsigned int phi,un_e, vn_e, n, L_r;
	unsigned long long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, D, E, F, A1, B1, C1, D1, x_s, y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;

	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;

	adresse_une_vne=(unsigned long long int)(un_e+(vn_e-vn_start)*blockDim.x*gridDim.x)+(unsigned long long int)(phi)*(unsigned long long int)(blockDim.x*gridDim.x)*(unsigned long long int)(blockDim.y*gridDim.y);

	//Coord de la source et du detecteur
	x_s = -focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
	y_s = -focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
	A = alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) - ((float)un_e - uDetectorCenterPixel_GPU) * betaIOcylinderC_GPU[phi]*uDetectorPixelSize_GPU; //x_det
	B = betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) + ((float)un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi]; //y_det
	C = ((float)vn_e - vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU; //z_det

	//Calcul de la longueur du rayon
	D = 1.0 / sqrtf((A - x_s)*(A - x_s) + (B - y_s)*(B - y_s) + C*C); //L
	salpha = C*D;
	//	salpha = C*D;
	ctheta = (A - x_s)*D;
	stheta = (B - y_s)*D;

	D = 1 / (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) + alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU));
	A1 = (alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) - (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)))*D;
	B1 = ((un_e - uDetectorCenterPixel_GPU)*focusObjectDistance_GPU)*D;
	C1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU + A1*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU);
	D1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*B1 + focusObjectDistance_GPU);

	F = ((float)xVolumePixelNb_GPU)/2.0*((float)xVolumePixelNb_GPU)/2.0*(A1*A1 + 1) - B1*B1;

	if(F >= 0)
	{

		E = sqrtf(F);
		x_p1 = (-A1*B1 + E) / (1 + A1*A1);
		y_p1 = A1*x_p1 + B1;
		z_p1 = C1*x_p1 + D1;
		x_p2 = (-A1*B1 - E) / (1 + A1*A1);
		y_p2 = A1*x_p2 + B1;
		z_p2 = C1*x_p2 + D1;

		//Calcul de distances a la source
		A = (x_p1 - x_s) * (x_p1 - x_s) + (y_p1 - y_s) * (y_p1 - y_s) + (z_p1) * (z_p1); //d_M1_s
		B = (x_p2 - x_s) * (x_p2 - x_s) + (y_p2 - y_s) * (y_p2 - y_s) + (z_p2) * (z_p2); //d_M2_s

		//Calcul de (x_i,y_i) et (x_f,y_f)
		if(A <= B)
		{
			A = x_p1 + xVolumeCenterPixel_GPU; //x_i
			B = y_p1 + yVolumeCenterPixel_GPU; //y_i
			C = z_p1 + zVolumeCenterPixel_GPU; //z_i

			x_f = x_p2 + xVolumeCenterPixel_GPU;
			y_f = y_p2 + yVolumeCenterPixel_GPU;
			z_f = z_p2 + zVolumeCenterPixel_GPU;
		}
		else
		{
			A = x_p2 + xVolumeCenterPixel_GPU; //x_i
			B = y_p2 + yVolumeCenterPixel_GPU; //y_i
			C = z_p2 + zVolumeCenterPixel_GPU; //z_i

			x_f = x_p1 + xVolumeCenterPixel_GPU;
			y_f = y_p1 + yVolumeCenterPixel_GPU;
			z_f = z_p1 + zVolumeCenterPixel_GPU;
		}

		//if (C < float(zVolumePixelNb_GPU-1) && C > 0)
		if (C < float(zVolumePixelNb_GPU) && C >= 0)
		{
			if (z_f > float(zVolumePixelNb_GPU-1))
			{
				z_f = (zVolumePixelNb_GPU-1) - zVolumeCenterPixel_GPU;
				x_f = (z_f - D1) / C1;
				y_f = A1 * x_f + B1;

				x_f = x_f + xVolumeCenterPixel_GPU;
				y_f = y_f + yVolumeCenterPixel_GPU;
				z_f = z_f + zVolumeCenterPixel_GPU;
			}
			else if (z_f < 0.0)
			{
				z_f = - zVolumeCenterPixel_GPU;
				x_f = (z_f - D1) / C1;
				y_f = A1 * x_f + B1;

				x_f = x_f + xVolumeCenterPixel_GPU;
				y_f = y_f + yVolumeCenterPixel_GPU;
				z_f = z_f + zVolumeCenterPixel_GPU;
			}

			L_r = (unsigned long long int)(sqrtf((A - x_f) * (A - x_f) + (B - y_f) * (B - y_f) + (C - z_f) * (C - z_f)));
			x_s=0.0;

			C=C-zn_start;

			for (n = 0; n < L_r; n++)
			{
				x_s+=xVolumePixelSize_GPU*tex3D(volume_tex,A+0.5,B+0.5,C+0.5); //temp

				//Echantillonnage de delta_xn
				A += ctheta; //x_i
				B += stheta; //y_i
				C += salpha; //z_i
			}
			sinogram_g[adresse_une_vne]=__float2half_rn(x_s);
		}
		else
			sinogram_g[adresse_une_vne]=__float2half_rn(0);

	}
	else
		sinogram_g[adresse_une_vne]=__float2half_rn(0);


}



__global__ void
projection_ERB_kernel_v0_half(unsigned short *sinogram_g,int phi_start, int N_un_start, int N_vn_start, int N_zn_start,int N_un_sino_temp,int N_vn_sino_temp)
{
	unsigned int phi, adresse_phi,un_e, vn_e, n, L_r;
	unsigned long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, D, E, F, A1, B1, C1, D1, x_s, y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;

	un_e=threadIdx.x+blockIdx.x*blockDim.x+N_un_start;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+N_vn_start;

	phi=threadIdx.z + phi_start;
	adresse_phi=threadIdx.z;
	adresse_une_vne=(un_e-N_un_start)+(vn_e-N_vn_start)*N_un_sino_temp+adresse_phi*N_un_sino_temp*N_vn_sino_temp;

	//Coord de la source et du detecteur
	x_s = -focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
	y_s = -focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
	A = alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) - ((float)un_e - uDetectorCenterPixel_GPU) * betaIOcylinderC_GPU[phi]*uDetectorPixelSize_GPU; //x_det
	B = betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) + ((float)un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi]; //y_det
	C = ((float)vn_e - vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU; //z_det

	//Calcul de la longueur du rayon
	D = 1 / sqrtf((A - x_s)*(A - x_s) + (B - y_s)*(B - y_s) + C*C); //L
	salpha = C*D;
	ctheta = (A - x_s)*D;
	stheta = (B - y_s)*D;

	D = 1 / (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) + alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU));
	A1 = (alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) - (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)))*D;
	B1 = ((un_e - uDetectorCenterPixel_GPU)*focusObjectDistance_GPU)*D;
	C1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU + A1*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU);
	D1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*B1 + focusObjectDistance_GPU);

	F = ((float)xVolumePixelNb_GPU)/2.0*((float)xVolumePixelNb_GPU)/2.0*(A1*A1 + 1) - B1*B1;


	if(F >= 0)
	{

		E = sqrtf(F);
		x_p1 = (-A1 * B1 + E) / (1 + A1*A1);
		y_p1 = A1 * x_p1 + B1;
		z_p1 = C1 * x_p1 + D1;
		x_p2 = (-A1 * B1 - E) / (1 + A1*A1);
		y_p2 = A1 * x_p2 + B1;
		z_p2 = C1 * x_p2 + D1;

		//Calcul de distances a la source
		A = (x_p1 - x_s) * (x_p1 - x_s) + (y_p1 - y_s) * (y_p1 - y_s) + (z_p1) * (z_p1); //d_M1_s
		B = (x_p2 - x_s) * (x_p2 - x_s) + (y_p2 - y_s) * (y_p2 - y_s) + (z_p2) * (z_p2); //d_M2_s

		//Calcul de (x_i,y_i) et (x_f,y_f)
		if(A <= B)
		{
			A = x_p1 + xVolumeCenterPixel_GPU; //x_i
			B = y_p1 + yVolumeCenterPixel_GPU; //y_i
			C = z_p1 + zVolumeCenterPixel_GPU; //z_i

			x_f = x_p2 + xVolumeCenterPixel_GPU;
			y_f = y_p2 + yVolumeCenterPixel_GPU;
			z_f = z_p2 + zVolumeCenterPixel_GPU;
		}
		else
		{
			A = x_p2 + xVolumeCenterPixel_GPU; //x_i
			B = y_p2 + yVolumeCenterPixel_GPU; //y_i
			C = z_p2 + zVolumeCenterPixel_GPU; //z_i

			x_f = x_p1 + xVolumeCenterPixel_GPU;
			y_f = y_p1 + yVolumeCenterPixel_GPU;
			z_f = z_p1 + zVolumeCenterPixel_GPU;
		}

		//if (C < float(zVolumePixelNb_GPU-1) && C > 0)
		if (C < float(zVolumePixelNb_GPU) && C >= 0)
		{
			if (z_f > float(zVolumePixelNb_GPU-1))
			{
				z_f = (zVolumePixelNb_GPU-1) - zVolumeCenterPixel_GPU;
				x_f = (z_f - D1) / C1;
				y_f = A1 * x_f + B1;

				x_f = x_f + xVolumeCenterPixel_GPU;
				y_f = y_f + yVolumeCenterPixel_GPU;
				z_f = z_f + zVolumeCenterPixel_GPU;
			}
			else if (z_f < 0.0)
			{
				z_f = - zVolumeCenterPixel_GPU;
				x_f = (z_f - D1) / C1;
				y_f = A1 * x_f + B1;

				x_f = x_f + xVolumeCenterPixel_GPU;
				y_f = y_f + yVolumeCenterPixel_GPU;
				z_f = z_f + zVolumeCenterPixel_GPU;
			}

			L_r = (unsigned int)(sqrtf((A - x_f) * (A - x_f) + (B - y_f) * (B - y_f) + (C - z_f) * (C - z_f)));
			x_s=0;

			C=C-N_zn_start;

			for (n = 0; n < L_r; n++)
			{
				x_s+=xVolumePixelSize_GPU*tex3D(volume_tex,A+0.5,B+0.5,C+0.5); //temp

				//Echantillonnage de delta_xn
				A += ctheta; //x_i
				B += stheta; //y_i
				C += salpha; //z_i
			}
			sinogram_g[adresse_une_vne] = __float2half_rn((float)adresse_une_vne);
		}
		else
			sinogram_g[adresse_une_vne] = __float2half_rn(0);
	}
	else
		sinogram_g[adresse_une_vne] = __float2half_rn(0);

}
#endif // #ifndef _PROJECTION_ER_KERNEL_HALF_H_
