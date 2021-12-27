#ifndef _BACKPROJECTION_KERNEL_1_H_
#define _BACKPROJECTION_KERNEL_1_H_



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
__global__ void backprojection_VIB_kernel_v0_1reg(T *volume_g,int phi_start,int zn_start,int N_sLphi,int vn_start)
{
	int xn = threadIdx.x+blockIdx.x*blockDim.x ,yn = threadIdx.y+blockIdx.y*blockDim.y, offset_volume=xn+yn*xVolumePixelNb_GPU, a = (int)xVolumePixelNb_GPU*(int)yVolumePixelNb_GPU;

	float xn_prime = xn-xVolumeCenterPixel_GPU ,yn_prime = yn-yVolumeCenterPixel_GPU, zn_prime;



	if (4*((xn_prime)*(xn_prime)+(yn_prime)*(yn_prime)) < (xVolumePixelNb_GPU*xVolumePixelNb_GPU))
	{

		for (zn_prime=zn_start-zVolumeCenterPixel_GPU;zn_prime < zn_start-zVolumeCenterPixel_GPU+16;zn_prime++)
		{
			int phi = phi_start;
			int phi_layer=0;
			float valeur_dans_registre_0=0.0;


			float wn_inverse = 1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);
			float un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;

			float A=1.0/gammaIOcylinderC_GPU*wn_inverse;
			float vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;
			float vn_tex =vn-vn_start;


			valeur_dans_registre_0=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);



			phi_layer++;
			phi++;

			for (;phi<phi_start+N_sLphi;phi++)
			{
				wn_inverse=1.0f/(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*xn_prime+betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*yn_prime+focusObjectDistance_GPU);

				un=(-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*xn_prime+alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*yn_prime)*wn_inverse+uDetectorCenterPixel_GPU+0.5;
				A=1/gammaIOcylinderC_GPU*wn_inverse;
				vn=A*zn_prime+vDetectorCenterPixel_GPU+0.5;

				vn_tex=vn-vn_start;

				valeur_dans_registre_0+=tex2DLayered(sinogram_tex0,un,vn_tex,phi_layer);


				phi_layer++;
			}



			volume_g[offset_volume]=valeur_dans_registre_0*xVolumePixelSize_GPU;
			offset_volume+=a;


		}
	}
}

#endif // #ifndef _BACKPROJECTION_KERNEL_H_
