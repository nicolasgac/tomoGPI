#ifndef _BACKPROJECTION_KERNEL_UM_1_H_
#define _BACKPROJECTION_KERNEL_UM_1_H_



////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
/*! \fn __global__ void backprojection_VIB_kernel_v0_16reg(float *volume_g,int phi_start,int N_Lz,int N_ssLz,int N_sLphi,int N_xn_start,int N_yn_start,int N_zn_start,int N_vn_start)
  \brief Kernel du retroprojecteur pour un objet 3D sur multiGPU
  \param *volume_g Volume de sortie (GPU)
  \param phi_start Angle d'initialisation pour le kernel
  \param N_Lz
  \param N_ssLz
  \param N_sLphi
  \param N_xn_start xn d'initialisation pour definir le morceau du volume a calculer
  \param N_yn_start yn d'initialisation pour definir le morceau du volume a calculer
  \param N_zn_start zn d'initialisation pour definir le morceau du volume a calculer
  \param N_vn_start vn d'initialisation pour definir si on travaille dans le plan nord ou sud du plan detecteur
  \author Asier Rabanal
 */
template <typename T>
__global__ void backprojection_VIB_kernel_v0_1reg_UM(T *volume_g,int N_sLphi)
{
	unsigned long long int xn,yn,zn_start,a,offset_volume;
	float xn_prime,yn_prime,zn_prime;
	a=xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn_start=threadIdx.z+blockIdx.z*blockDim.z*16;

	offset_volume=xn+yn*xVolumePixelNb_GPU+zn_start*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	xn_prime=xn-xVolumeCenterPixel_GPU;
	yn_prime=yn-yVolumeCenterPixel_GPU;
#if FOV_CYLINDER
	if ((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)<((xVolumePixelNb_GPU)/2)*((xVolumePixelNb_GPU)/2))
	{
#endif


		for (zn_prime=zn_start-zVolumeCenterPixel_GPU;zn_prime<zn_start-zVolumeCenterPixel_GPU+16;zn_prime+=1)
		{
			int phi=0;
			int phi_layer=0;

			float valeur_dans_registre_0=0.0;

			float wn_inverse=0.0;
			float un=0.0;
			float A=0.0;
			float vn=0.0;

			wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

			un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
			A=1.0/gammaIOcylinderC_GPU*wn_inverse;
			vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;

			valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn,phi_layer);


			phi_layer++;
			phi++;

			for (;phi<0+N_sLphi;phi++)
			{
				wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

				un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
				A=1/gammaIOcylinderC_GPU*wn_inverse;
				vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;

				valeur_dans_registre_0+=tex2DLayered(sinogram_tex0,un,vn,phi_layer);

				phi_layer++;
			}

			volume_g[offset_volume]=valeur_dans_registre_0*xVolumePixelSize_GPU;
			offset_volume+=a;

		}
#if FOV_CYLINDER
	}
#endif

}

#endif // #ifndef _BACKPROJECTION_KERNEL_H_