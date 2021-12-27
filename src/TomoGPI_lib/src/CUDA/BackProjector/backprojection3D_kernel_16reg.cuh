#ifndef _BACKPROJECTION_KERNEL_16_H_
#define _BACKPROJECTION_KERNEL_16_H_



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
__global__ void backprojection_VIB_kernel_v1_16reg(T *volume_g,int phi_start,int zn_start,int N_sLphi,int vn_start)
{
	int xn,yn, offset_volume, a;
	float xn_prime,yn_prime;// zn_prime;

	a = (int)xVolumePixelNb_GPU*(int)yVolumePixelNb_GPU;

	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;

	offset_volume=xn+yn*xVolumePixelNb_GPU;

	xn_prime=xn-xVolumeCenterPixel_GPU;
	yn_prime=yn-yVolumeCenterPixel_GPU;

#if FOV_CYLINDER
	if ((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)<((xVolumePixelNb_GPU)/2)*((xVolumePixelNb_GPU)/2))
	{
#endif


		for (unsigned int zn=zn_start;zn<zn_start+16;zn+=16)
		{
			int phi = phi_start;
			int phi_layer=0;

			float valeur_dans_registre_0=0.0;
			float valeur_dans_registre_1=0.0;
			float valeur_dans_registre_2=0.0;
			float valeur_dans_registre_3=0.0;
			float valeur_dans_registre_4=0.0;
			float valeur_dans_registre_5=0.0;
			float valeur_dans_registre_6=0.0;
			float valeur_dans_registre_7=0.0;
			float valeur_dans_registre_8=0.0;
			float valeur_dans_registre_9=0.0;
			float valeur_dans_registre_10=0.0;
			float valeur_dans_registre_11=0.0;
			float valeur_dans_registre_12=0.0;
			float valeur_dans_registre_13=0.0;
			float valeur_dans_registre_14=0.0;
			float valeur_dans_registre_15=0.0;

			float wn_inverse=0.0;
			float un=0.0;
			float A=0.0;
			float vn=0.0;
			float vn_tex;

			wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

			un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
			A=1.0/gammaIOcylinderC_GPU*wn_inverse;
			vn=A*((float)zn-zVolumeCenterPixel_GPU)+vDetectorCenterPixel_GPU+0.5f;
			vn=vn-vn_start;

			valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn,phi_layer);
			valeur_dans_registre_1=tex2DLayered(sinogram_tex0,un,vn+A,phi_layer);
			valeur_dans_registre_2=tex2DLayered(sinogram_tex0,un,vn+2*A,phi_layer);
			valeur_dans_registre_3=tex2DLayered(sinogram_tex0,un,vn+3*A,phi_layer);
			valeur_dans_registre_4=tex2DLayered(sinogram_tex0,un,vn+4*A,phi_layer);
			valeur_dans_registre_5=tex2DLayered(sinogram_tex0,un,vn+5*A,phi_layer);
			valeur_dans_registre_6=tex2DLayered(sinogram_tex0,un,vn+6*A,phi_layer);
			valeur_dans_registre_7=tex2DLayered(sinogram_tex0,un,vn+7*A,phi_layer);
			valeur_dans_registre_8=tex2DLayered(sinogram_tex0,un,vn+8*A,phi_layer);
			valeur_dans_registre_9=tex2DLayered(sinogram_tex0,un,vn+9*A,phi_layer);
			valeur_dans_registre_10=tex2DLayered(sinogram_tex0,un,vn+10*A,phi_layer);
			valeur_dans_registre_11=tex2DLayered(sinogram_tex0,un,vn+11*A,phi_layer);
			valeur_dans_registre_12=tex2DLayered(sinogram_tex0,un,vn+12*A,phi_layer);
			valeur_dans_registre_13=tex2DLayered(sinogram_tex0,un,vn+13*A,phi_layer);
			valeur_dans_registre_14=tex2DLayered(sinogram_tex0,un,vn+14*A,phi_layer);
			valeur_dans_registre_15=tex2DLayered(sinogram_tex0,un,vn+15*A,phi_layer);
			vn+=15*A;

			phi_layer++;
			phi++;

			for (;phi<phi_start+N_sLphi;phi++)
			{
				wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

				un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
				A=1/gammaIOcylinderC_GPU*wn_inverse;
				vn=A*((float)zn-zVolumeCenterPixel_GPU)+vDetectorCenterPixel_GPU+0.5f;

				valeur_dans_registre_0+=tex2DLayered(sinogram_tex0,un,vn,phi_layer);
				valeur_dans_registre_1+=tex2DLayered(sinogram_tex0,un,vn+A,phi_layer);
				valeur_dans_registre_2+=tex2DLayered(sinogram_tex0,un,vn+2*A,phi_layer);
				valeur_dans_registre_3+=tex2DLayered(sinogram_tex0,un,vn+3*A,phi_layer);
				valeur_dans_registre_4+=tex2DLayered(sinogram_tex0,un,vn+4*A,phi_layer);
				valeur_dans_registre_5+=tex2DLayered(sinogram_tex0,un,vn+5*A,phi_layer);
				valeur_dans_registre_6+=tex2DLayered(sinogram_tex0,un,vn+6*A,phi_layer);
				valeur_dans_registre_7+=tex2DLayered(sinogram_tex0,un,vn+7*A,phi_layer);
				valeur_dans_registre_8+=tex2DLayered(sinogram_tex0,un,vn+8*A,phi_layer);
				valeur_dans_registre_9+=tex2DLayered(sinogram_tex0,un,vn+9*A,phi_layer);
				valeur_dans_registre_10+=tex2DLayered(sinogram_tex0,un,vn+10*A,phi_layer);
				valeur_dans_registre_11+=tex2DLayered(sinogram_tex0,un,vn+11*A,phi_layer);
				valeur_dans_registre_12+=tex2DLayered(sinogram_tex0,un,vn+12*A,phi_layer);
				valeur_dans_registre_13+=tex2DLayered(sinogram_tex0,un,vn+13*A,phi_layer);
				valeur_dans_registre_14+=tex2DLayered(sinogram_tex0,un,vn+14*A,phi_layer);
				valeur_dans_registre_15+=tex2DLayered(sinogram_tex0,un,vn+15*A,phi_layer);


				phi_layer++;
			}



			volume_g[offset_volume]=valeur_dans_registre_0*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_1*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_2*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_3*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_4*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_5*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_6*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_7*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_8*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_9*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_10*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_11*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_12*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_13*xVolumePixelSize_GPU;
			offset_volume+=a;

			volume_g[offset_volume]=valeur_dans_registre_14*xVolumePixelSize_GPU;
			offset_volume+=a;


			volume_g[offset_volume]=valeur_dans_registre_15*xVolumePixelSize_GPU;
			offset_volume+=a;
		}
#if FOV_CYLINDER
	}
#endif

}

template <typename T>
__global__ void debug_back(T *volume_g,int phi_start,int zn_start,int N_sLphi,int vn_start)
{
	int offset_volume;

	float xn_prime,yn_prime;//, zn_prime;


	xn_prime=threadIdx.x+blockIdx.x*blockDim.x;
	yn_prime=threadIdx.y+blockIdx.y*blockDim.y;

	offset_volume=(int)xn_prime+(int)yn_prime*xVolumePixelNb_GPU;

	xn_prime=xn_prime-xVolumeCenterPixel_GPU;
	yn_prime=yn_prime-yVolumeCenterPixel_GPU;


	for (int zn=0;zn<16;zn+=16)
	{
		int phi = phi_start;
		int phi_layer=0;

		float valeur_dans_registre_0=0;
		float valeur_dans_registre_1=0;
		float valeur_dans_registre_2=0;
		float valeur_dans_registre_3=0;
		float valeur_dans_registre_4=0;
		float valeur_dans_registre_5=0;
		float valeur_dans_registre_6=0;
		float valeur_dans_registre_7=0;
		float valeur_dans_registre_8=0;
		float valeur_dans_registre_9=0;
		float valeur_dans_registre_10=0;
		float valeur_dans_registre_11=0;
		float valeur_dans_registre_12=0;
		float valeur_dans_registre_13=0;
		float valeur_dans_registre_14=0;
		float valeur_dans_registre_15=0;

		//float wn_inverse=0.0;
		float un=0.0;
		float A=0.0;
		//float vn=0.0;
		float vn_tex=0.0;

		A=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);//wn_inverse=....

		un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*A+uDetectorCenterPixel_GPU+0.5f;
		A=1.0/gammaIOcylinderC_GPU*A;//A=1.0/gammaIOcylinderC_GPU*wn_inverse;
		vn_tex=A*(zn+zn_start-zVolumeCenterPixel_GPU)+vDetectorCenterPixel_GPU+0.5f;
		vn_tex=vn_tex-vn_start;

		valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_1=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_2=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_3=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_4=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_5=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_6=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_7=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_8=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_9=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_10=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_11=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_12=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_13=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_14=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		vn_tex+=A;
		valeur_dans_registre_15=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

		phi_layer++;
		phi++;

		{
			int a ;

			a = (int)xVolumePixelNb_GPU*(int)yVolumePixelNb_GPU;


			volume_g[offset_volume]=2;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=8;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=4;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=5;
			offset_volume+=a;

			volume_g[offset_volume]=5;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=8;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=3;
			offset_volume+=a;

			volume_g[offset_volume]=2;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;

			volume_g[offset_volume]=1;
			offset_volume+=a;
		}
	}



}

template <typename T>
__global__ void backprojection_VIB_kernel_v2_16reg(T *volume_g,int phi_start,int zn_start,int N_sLphi,int vn_start)
{
	//int xn,yn;
	int offset_volume;

	float xn_prime,yn_prime;//, zn_prime;


	xn_prime=threadIdx.x+blockIdx.x*blockDim.x;
	yn_prime=threadIdx.y+blockIdx.y*blockDim.y;

	offset_volume=(int)xn_prime+(int)yn_prime*xVolumePixelNb_GPU;

	xn_prime=xn_prime-xVolumeCenterPixel_GPU;
	yn_prime=yn_prime-yVolumeCenterPixel_GPU;



#if FOV_CYLINDER
	if ((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)<((xVolumePixelNb_GPU)/2)*((xVolumePixelNb_GPU)/2))
	{
#endif

		for (int zn=0;zn<16;zn+=16)
		{
			int phi = phi_start;
			int phi_layer=0;

			float valeur_dans_registre_0=0;
			float valeur_dans_registre_1=0;
			float valeur_dans_registre_2=0;
			float valeur_dans_registre_3=0;
			float valeur_dans_registre_4=0;
			float valeur_dans_registre_5=0;
			float valeur_dans_registre_6=0;
			float valeur_dans_registre_7=0;
			float valeur_dans_registre_8=0;
			float valeur_dans_registre_9=0;
			float valeur_dans_registre_10=0;
			float valeur_dans_registre_11=0;
			float valeur_dans_registre_12=0;
			float valeur_dans_registre_13=0;
			float valeur_dans_registre_14=0;
			float valeur_dans_registre_15=0;

			//float wn_inverse=0.0;
			float un=0.0;
			float A=0.0;
			//float vn=0.0;
			float vn_tex=0.0;

			A=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);//wn_inverse=....

			un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*A+uDetectorCenterPixel_GPU+0.5f;
			A=1.0/gammaIOcylinderC_GPU*A;//A=1.0/gammaIOcylinderC_GPU*wn_inverse;
			vn_tex=A*(zn+zn_start-zVolumeCenterPixel_GPU)+vDetectorCenterPixel_GPU+0.5f;
			vn_tex=vn_tex-vn_start;

			valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_1=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_2=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_3=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_4=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_5=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_6=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_7=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_8=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_9=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_10=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_11=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_12=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_13=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_14=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			vn_tex+=A;
			valeur_dans_registre_15=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

			phi_layer++;
			phi++;

			for (;phi<phi_start+N_sLphi;phi++)
			{

				A=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

				un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*A+uDetectorCenterPixel_GPU+0.5f;
				A=1/gammaIOcylinderC_GPU*A;//A=1/gammaIOcylinderC_GPU*wn_inverse
				vn_tex=A*(zn+zn_start-zVolumeCenterPixel_GPU)+vDetectorCenterPixel_GPU+0.5f;

				vn_tex=vn_tex-vn_start;

				valeur_dans_registre_0+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_1+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_2+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_3+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_4+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_5+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_6+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_7+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_8+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_9+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_10+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_11+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_12+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_13+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_14+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				vn_tex+=A;
				valeur_dans_registre_15+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);

				phi_layer++;
			}
			{
				int a ;

				a = (int)xVolumePixelNb_GPU*(int)yVolumePixelNb_GPU;


				volume_g[offset_volume]=valeur_dans_registre_0*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_1*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_2*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_3*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_4*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_5*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_6*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_7*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_8*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_9*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_10*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_11*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_12*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_13*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_14*xVolumePixelSize_GPU;
				offset_volume+=a;

				volume_g[offset_volume]=valeur_dans_registre_15*xVolumePixelSize_GPU;
				offset_volume+=a;
			}
		}
#if FOV_CYLINDER
	}
#endif
}

#endif
