#ifndef _BACKPROJECTION_KERNEL_UM_8_H_
#define _BACKPROJECTION_KERNEL_UM_8_H_



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
__global__ void backprojection_VIB_kernel_v1_8reg_UM(T *volume_g,int N_sLphi)
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

	if ((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)<((xVolumePixelNb_GPU)/2)*((xVolumePixelNb_GPU)/2))
	{

		for (zn_prime=zn_start-zVolumeCenterPixel_GPU;zn_prime<zn_start-zVolumeCenterPixel_GPU+16;zn_prime+=8)
		{
			int phi=0;
			int phi_layer=0;

			float valeur_dans_registre_0=0.0;
			float valeur_dans_registre_1=0.0;
			float valeur_dans_registre_2=0.0;
			float valeur_dans_registre_3=0.0;
			float valeur_dans_registre_4=0.0;
			float valeur_dans_registre_5=0.0;
			float valeur_dans_registre_6=0.0;
			float valeur_dans_registre_7=0.0;

			float wn_inverse=0.0;
			float un=0.0;
			float A=0.0;
			float vn=0.0;

			wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

			un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
			A=1.0/gammaIOcylinderC_GPU*wn_inverse;
			vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;


			valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn,phi_layer);
			valeur_dans_registre_1=tex2DLayered(sinogram_tex0,un,vn+A,phi_layer);
			valeur_dans_registre_2=tex2DLayered(sinogram_tex0,un,vn+2*A,phi_layer);
			valeur_dans_registre_3=tex2DLayered(sinogram_tex0,un,vn+3*A,phi_layer);
			valeur_dans_registre_4=tex2DLayered(sinogram_tex0,un,vn+4*A,phi_layer);
			valeur_dans_registre_5=tex2DLayered(sinogram_tex0,un,vn+5*A,phi_layer);
			valeur_dans_registre_6=tex2DLayered(sinogram_tex0,un,vn+6*A,phi_layer);
			valeur_dans_registre_7=tex2DLayered(sinogram_tex0,un,vn+7*A,phi_layer);

			vn+=7*A;

			phi_layer++;
			phi++;

			for (;phi<0+N_sLphi;phi++)
			{
				wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);
				un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
				A=1/gammaIOcylinderC_GPU*wn_inverse;
				vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;

				valeur_dans_registre_0+=tex2DLayered(sinogram_tex0,un,vn,phi_layer);
				valeur_dans_registre_1+=tex2DLayered(sinogram_tex0,un,vn+A,phi_layer);
				valeur_dans_registre_2+=tex2DLayered(sinogram_tex0,un,vn+2*A,phi_layer);
				valeur_dans_registre_3+=tex2DLayered(sinogram_tex0,un,vn+3*A,phi_layer);
				valeur_dans_registre_4+=tex2DLayered(sinogram_tex0,un,vn+4*A,phi_layer);
				valeur_dans_registre_5+=tex2DLayered(sinogram_tex0,un,vn+5*A,phi_layer);
				valeur_dans_registre_6+=tex2DLayered(sinogram_tex0,un,vn+6*A,phi_layer);
				valeur_dans_registre_7+=tex2DLayered(sinogram_tex0,un,vn+7*A,phi_layer);
				phi_layer++;

			}

			volume_g[offset_volume]=valeur_dans_registre_0*xVolumePixelSize_GPU;
			volume_g[offset_volume+a]=valeur_dans_registre_1*xVolumePixelSize_GPU;
			volume_g[offset_volume+2*a]=valeur_dans_registre_2*xVolumePixelSize_GPU;
			volume_g[offset_volume+3*a]=valeur_dans_registre_3*xVolumePixelSize_GPU;
			volume_g[offset_volume+4*a]=valeur_dans_registre_4*xVolumePixelSize_GPU;
			volume_g[offset_volume+5*a]=valeur_dans_registre_5*xVolumePixelSize_GPU;
			volume_g[offset_volume+6*a]=valeur_dans_registre_6*xVolumePixelSize_GPU;
			volume_g[offset_volume+7*a]=valeur_dans_registre_7*xVolumePixelSize_GPU;

			offset_volume+=8*a;
		}
	}
}

#endif // #ifndef _BACKPROJECTION_KERNEL_H_
