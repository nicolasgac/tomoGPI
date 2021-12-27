#ifndef _FDK3D_KERNEL_H_
#define _FDK3D_KERNEL_H_



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
__global__ void FDK3D_VIB_kernel_v0_16reg(T *volume_g,int phi_start,int zn_start,int N_sLphi,int vn_start)
{
	int xn,yn,offset_volume;
	volatile float xn_prime,yn_prime,zn_prime;
	int a = (int)xVolumePixelNb_GPU*(int)yVolumePixelNb_GPU;

	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;

	offset_volume=xn+yn*xVolumePixelNb_GPU;
	xn_prime=xn-xVolumeCenterPixel_GPU;
	yn_prime=yn-yVolumeCenterPixel_GPU;

#if FOV_CYLINDER
	if ((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)<(xVolumePixelNb_GPU/2)*(xVolumePixelNb_GPU/2))
	{
#endif

		for (zn_prime=zn_start-zVolumeCenterPixel_GPU;zn_prime<zn_start-zVolumeCenterPixel_GPU+16;zn_prime+=16)
		{
			int phi;

			volatile float valeur_dans_registre_0=0;
			volatile float valeur_dans_registre_1=0;
			volatile float valeur_dans_registre_2=0;
			volatile float valeur_dans_registre_3=0;
			volatile float valeur_dans_registre_4=0;
			volatile float valeur_dans_registre_5=0;
			volatile float valeur_dans_registre_6=0;
			volatile float valeur_dans_registre_7=0;
			volatile float valeur_dans_registre_8=0;
			volatile float valeur_dans_registre_9=0;
			volatile float valeur_dans_registre_10=0;
			volatile float valeur_dans_registre_11=0;
			volatile float valeur_dans_registre_12=0;
			volatile float valeur_dans_registre_13=0;
			volatile float valeur_dans_registre_14=0;
			volatile float valeur_dans_registre_15=0;

			phi=phi_start;
			int phi_layer=0;

			float wn_inverse;
			float un,un_prime;
			float A;
			float vn,vn_prime,vn_tex;

			wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

			un_prime=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse;
			un=un_prime+uDetectorCenterPixel_GPU+0.5;
			A=(1.0f/gammaIOcylinderC_GPU)*wn_inverse;
			vn_prime=A*zn_prime;
			vn=vn_prime+vDetectorCenterPixel_GPU+0.5;
			vn_tex=vn-vn_start;

			float denum,fact_FDK,fact_FDK_num,un_vn_prime_carre,un_prime_carre;
			denum=(xn_prime*alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU+yn_prime*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU+focusObjectDistance_GPU);
			fact_FDK_num=alphaC_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)/(denum*denum);
			un_prime_carre=un_prime*un_prime;

			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);

			valeur_dans_registre_0=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_1=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_2=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_3=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_4=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_5=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_6=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_7=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_8=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_9=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_10=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_11=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_12=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_13=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_14=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			vn_prime+=A;
			un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
			fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
			valeur_dans_registre_15=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			phi_layer++;
			phi++;

			for (;phi<phi_start+N_sLphi;phi++)
			{
				wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

				un_prime=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse;
				un=un_prime+uDetectorCenterPixel_GPU+0.5;
				A=(1.0f/gammaIOcylinderC_GPU)*wn_inverse;
				vn_prime=A*zn_prime;
				vn=vn_prime+vDetectorCenterPixel_GPU+0.5;

				vn_tex=vn-vn_start;

				float denum,fact_FDK,fact_FDK_num,un_vn_prime_carre,un_prime_carre;
				denum=(xn_prime*alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU+yn_prime*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU+focusObjectDistance_GPU);
				fact_FDK_num=alphaC_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)/(denum*denum);
				un_prime_carre=un_prime*un_prime;

				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_0+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_1+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_2+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_3+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_4+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_5+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_6+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_7+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_8+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_9+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_10+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_11+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_12+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_13+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_14+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				vn_prime+=A;
				un_vn_prime_carre=un_prime_carre+vn_prime*vn_prime;
				fact_FDK=fact_FDK_num/sqrtf(betaC_GPU+un_vn_prime_carre);
				valeur_dans_registre_15+=fact_FDK*tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);
				phi_layer++;
			}

			volume_g[offset_volume]=valeur_dans_registre_0;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_1;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_2;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_3;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_4;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_5;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_6;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_7;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_8;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_9;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_10;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_11;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_12;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_13;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_14;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_15;
			offset_volume+=a;
		}
#if FOV_CYLINDER
	}
#endif
}

#endif // #ifndef _BACKPROJECTION_FDK_KERNEL_H_
