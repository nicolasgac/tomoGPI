#ifndef _PROJECTION_ER_KERNEL_H_
#define _PROJECTION_ER_KERNEL_H_



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

template <typename T>
__global__ void projection_debug(T *sinogram_g,int vn_start, int zn_start)
{
	unsigned int phi,un_e, vn_e, L_r;
	unsigned long long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, E, F, A1, B1, C1, D1, x_s,y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;
	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;

	adresse_une_vne=(unsigned long int)(un_e+(vn_e-vn_start)*blockDim.x*gridDim.x)+(unsigned long long int)(phi)*(unsigned long long int)(blockDim.x*gridDim.x)*(unsigned long long int)(blockDim.y*gridDim.y);

	if ((zn_start>=0)&& (zn_start<=61))
	{
		sinogram_g[adresse_une_vne]=1.0f;
	}
	else if ((zn_start>=62)&& (zn_start<=126))
	{
		sinogram_g[adresse_une_vne]=0.25f;
	}
	else if ((zn_start>=127)&& (zn_start<=189))
	{
		sinogram_g[adresse_une_vne]=0.5f;
	}
	else if ((zn_start>=190)&& (zn_start<=255))
	{
		sinogram_g[adresse_une_vne]=0.75f;
	}
}


template <typename T>
__global__ void projection_ERB_kernel_v1(T *sinogram_g,int vn_start, int zn_start){
	unsigned int phi,un_e, vn_e, L_r;
	unsigned long long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, E, F, A1, B1, C1, D1, x_s,y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;
	//float D;
	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;

	adresse_une_vne=(unsigned long int)(un_e+(vn_e-vn_start)*blockDim.x*gridDim.x)+(unsigned long long int)(phi)*(unsigned long long int)(blockDim.x*gridDim.x)*(unsigned long long int)(blockDim.y*gridDim.y);

	//Coord de la source et du detecteur
	x_s = -focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
	y_s = -focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
	A = alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) - ((float)un_e - uDetectorCenterPixel_GPU) * betaIOcylinderC_GPU[phi]*uDetectorPixelSize_GPU; //x_det
	B = betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) + ((float)un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi]; //y_det
	C = ((float)vn_e - vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU; //z_det

	//Calcul de la longueur du rayon
	E = 1.0f / sqrtf((A - x_s)*(A - x_s) + (B - y_s)*(B - y_s) + C*C); //L
	salpha = C*E;
	//	salpha = C*E;
	ctheta = (A - x_s)*E;
	stheta = (B - y_s)*E;

	//A1=(B-y_s)/(A-x_s);
	//B1=y_s-A1*x_s;
	//C1=C/(A-x_s);
	//D1=-C1*A;

	/*D = 1.0f / (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*((float)un_e - uDetectorCenterPixel_GPU) + alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU));
	A1 = (alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*((float)un_e - uDetectorCenterPixel_GPU) - (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)))*D;
	B1 = (((float)un_e - uDetectorCenterPixel_GPU)*focusObjectDistance_GPU)*D;
	C1 = gammaIOcylinderC_GPU*((float)vn_e - vDetectorCenterPixel_GPU)*(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU + A1*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU);
	D1 = gammaIOcylinderC_GPU*((float)vn_e - vDetectorCenterPixel_GPU)*(betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*B1 + focusObjectDistance_GPU);*/
	A1=(B-y_s)/(A-x_s); // (y_det-y_s)/(x_det-x_s)
	B1=(B-A1*A)/xVolumePixelSize_GPU; //(y_det-x_det*A1)/xVolumePixelSize_GPU
	C1=C/(A-x_s);//z_det/(x_det-x_s)
	D1=(C-C1*A)/xVolumePixelSize_GPU; //z_det-C1*x_det


#if FOV_CYLINDER
	F = ((float)xVolumePixelNb_GPU)/2.0f*((float)xVolumePixelNb_GPU)/2.0*(A1*A1 + 1.0f) - B1*B1;
#else
	//F = ((float)((uDetectorPixelSize_GPU*uDetectorPixelNb_GPU*focusObjectDistance_GPU/focusDetectorDistance_GPU)/xVolumePixelSize_GPU))/2.0f*((float)((vDetectorPixelSize_GPU*vDetectorPixelNb_GPU*focusObjectDistance_GPU/focusDetectorDistance_GPU)/xVolumePixelSize_GPU))/2.0*(A1*A1 + 1) - B1*B1;// le cercle qui englobe le carré
	F = 2.0f*((float)xVolumePixelNb_GPU)/2.0f*((float)xVolumePixelNb_GPU)/2.0*(A1*A1 + 1.0f) - B1*B1;
#endif



	if(F >= 0.0f){
		E = sqrtf(F);
		x_p1 = (-A1*B1 + E) / (1.0f + A1*A1);
		y_p1 = A1*x_p1 + B1;
		z_p1 = C1*x_p1 + D1;
		x_p2 = (-A1*B1 - E) / (1.0f + A1*A1);
		y_p2 = A1*x_p2 + B1;
		z_p2 = C1*x_p2 + D1;

		//Calcul de distances a la source
		A = (x_p1 - x_s) * (x_p1 - x_s) + (y_p1 - y_s) * (y_p1 - y_s) + (z_p1) * (z_p1); //d_M1_s
		B = (x_p2 - x_s) * (x_p2 - x_s) + (y_p2 - y_s) * (y_p2 - y_s) + (z_p2) * (z_p2); //d_M2_s

#if FOV_CYLINDER

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



#else
		//if (((A < 0.0f)&&(B < 0.0f))||((A < 0.0f)&&(B < 0.0f))){sinogram_g[adresse_une_vne]=0.0f;}
		//else {

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

		if (A < 0.0f){
			A=0.0f;
			B = A1*(A-xVolumeCenterPixel_GPU) + B1;
			C = C1*(A-xVolumeCenterPixel_GPU) + D1;

			B+=yVolumeCenterPixel_GPU;
			C+=zVolumeCenterPixel_GPU;
		}
		if (B < 0.0f){
			B=0.0f;
			A = (B-yVolumeCenterPixel_GPU-B1)/A1;
			C = C1*A + D1;

			A+=xVolumeCenterPixel_GPU;
			C+=zVolumeCenterPixel_GPU;

		}
		if (A >((float)(xVolumePixelNb_GPU-1)) ){
			A=(xVolumePixelNb_GPU-1);
			B = A1*(A-xVolumeCenterPixel_GPU) + B1;
			C = C1*(A-xVolumeCenterPixel_GPU) + D1;

			B+=yVolumeCenterPixel_GPU;
			C+=zVolumeCenterPixel_GPU;
		}
		if (B >((float)(yVolumePixelNb_GPU-1)) ){
			B=(yVolumePixelNb_GPU-1);
			A = (B-yVolumeCenterPixel_GPU-B1)/A1;
			C = C1*A + D1;

			A+=xVolumeCenterPixel_GPU;
			C+=zVolumeCenterPixel_GPU;

		}

		if (x_f<0.0f){
			x_f=0.0f;
			y_f = A1*(x_f-xVolumeCenterPixel_GPU) + B1;
			z_f = C1*(x_f-xVolumeCenterPixel_GPU) + D1;

			y_f+=yVolumeCenterPixel_GPU;
			z_f+=zVolumeCenterPixel_GPU;

		}
		if (y_f< 0.0f){
			y_f=0.0f;
			x_f = (y_f-yVolumeCenterPixel_GPU-B1)/A1;
			z_f = C1*x_f + D1;

			x_f+=xVolumeCenterPixel_GPU;
			z_f+=zVolumeCenterPixel_GPU;

		}
		if (x_f> ((float)(xVolumePixelNb_GPU-1))){
			x_f=(xVolumePixelNb_GPU-1);
			y_f = A1*(x_f-xVolumeCenterPixel_GPU) + B1;
			z_f = C1*(x_f-xVolumeCenterPixel_GPU) + D1;

			y_f+=yVolumeCenterPixel_GPU;
			z_f+=zVolumeCenterPixel_GPU;

		}
		if (y_f> ((float)(yVolumePixelNb_GPU-1))){
			y_f=(yVolumePixelNb_GPU-1);
			x_f = (y_f-yVolumeCenterPixel_GPU-B1)/A1;
			z_f = C1*x_f + D1;

			x_f+=xVolumeCenterPixel_GPU;
			z_f+=zVolumeCenterPixel_GPU;

		}



#endif

			//if (C < float(zVolumePixelNb_GPU-1) && C > 0)
			if ((C <= float(zVolumePixelNb_GPU-1)) && (C >= 0.0f))
			{
				float x_s=0.0f;

				if (z_f > float(zVolumePixelNb_GPU-1))
				{
					z_f = (zVolumePixelNb_GPU-1) - zVolumeCenterPixel_GPU;
					x_f = (z_f - D1) / C1;
					y_f = A1 * x_f + B1;

					x_f += xVolumeCenterPixel_GPU;
					y_f += yVolumeCenterPixel_GPU;
					z_f += zVolumeCenterPixel_GPU;
				}
				else if (z_f < 0.0)
				{
					z_f = - zVolumeCenterPixel_GPU;
					x_f = (z_f - D1) / C1;
					y_f = A1 * x_f + B1;

					x_f += xVolumeCenterPixel_GPU;
					y_f += yVolumeCenterPixel_GPU;
					z_f += zVolumeCenterPixel_GPU;
				}


				L_r = (unsigned long long int)(sqrtf((A - x_f) * (A - x_f) + (B - y_f) * (B - y_f) + (C - z_f) * (C - z_f)));


				C=C-zn_start;

				for (unsigned int n = 0; n < L_r; n++)
				{
					x_s+=xVolumePixelSize_GPU*tex3D(volume_tex,A+0.5f,B+0.5f,C+0.5f); //temp

					//Echantillonnage de delta_xn
					A += ctheta; //x_i
					B += stheta; //y_i
					C += salpha; //z_i
				}
				sinogram_g[adresse_une_vne]=x_s;
			}
			else
				sinogram_g[adresse_une_vne]=0.0f;

		}


//#ifndef FOV_CYLINDER
//	}
//#endif
	else
		sinogram_g[adresse_une_vne]=0.0f;
}

template <typename T>
__global__ void projection_ERB_kernel_v0(T *sinogram_g,int phi_start, int N_un_start, int N_vn_start, int N_zn_start,int N_un_sino_temp,int N_vn_sino_temp)
{
	unsigned long long int phi,un_e, vn_e, n, L_r;
	unsigned long long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, D, E, F, A1, B1, C1, D1, x_s, y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;

	un_e=threadIdx.x+blockIdx.x*blockDim.x+N_un_start;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+N_vn_start;

	phi=threadIdx.z + phi_start;
	adresse_une_vne=(un_e-N_un_start)+(vn_e-N_vn_start)*N_un_sino_temp+threadIdx.z*N_un_sino_temp*N_vn_sino_temp;



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
			C=C-N_zn_start;

			//			float tmp = 0.0;
			//			float tmp2 = C;
			//			float tmp3 = 0.0;


			for (n = 0; n < L_r; n++)
			{
				x_s += xVolumePixelSize_GPU*tex3D(volume_tex,A+0.5f,B+0.5f,C+0.5f);
				//				x_s= x_s + tmp3;

				//Echantillonnage de delta_xn
				A+=ctheta; //x_i
				B+=stheta; //y_i
				C+=salpha;
				//				tmp = (n+1)*salpha; //z_i
				//				tmp2 = C + tmp;
			}
			sinogram_g[adresse_une_vne]=x_s;
		}
		else
			sinogram_g[adresse_une_vne]=0.0;
	}
	else
		sinogram_g[adresse_une_vne]=0.0;

}
//#endif // #ifndef _PROJECTION_ER_KERNEL_H_

//reconstruction du cube
template <typename T>
__global__ void projection_ERB_kernel_v2(T *sinogram_g,int vn_start, int zn_start)
{
	unsigned int phi,un_e, vn_e, n, L_r;
	unsigned long long int adresse_une_vne;
	float x_f, y_f, z_f, A, B, C, D, E, F, A1, B1, C1, D1, x_s, y_s, x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, ctheta, stheta, salpha;

	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;

	adresse_une_vne=(unsigned long long int)(un_e+(vn_e-vn_start)*blockDim.x*gridDim.x)+(unsigned long long int)(phi)*(unsigned long long int)(blockDim.x*gridDim.x)*(unsigned long long int)(blockDim.y*gridDim.y);

	//Coord de la source
	x_s = -focusObjectDistance_GPU*alphaIOcylinderC_GPU[phi];
	y_s = -focusObjectDistance_GPU*betaIOcylinderC_GPU[phi];
	//z = 0

	//Coord du detecteur
	A = alphaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) - ((float)un_e - uDetectorCenterPixel_GPU) * betaIOcylinderC_GPU[phi]*uDetectorPixelSize_GPU; //x_det
	B = betaIOcylinderC_GPU[phi]*(focusDetectorDistance_GPU-focusObjectDistance_GPU) + ((float)un_e - uDetectorCenterPixel_GPU) * uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi]; //y_det
	C = ((float)vn_e - vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU; //z_det

	//Calcul de la longueur du rayon
	D = 1.0 / sqrtf((A - x_s)*(A - x_s) + (B - y_s)*(B - y_s) + C*C); //L
	salpha = C*D;

	//	salpha = C*D;
	ctheta = (A - x_s)*D;
	stheta = (B - y_s)*D;

	//A1=(B-y_s)/(A-x_s);
	//B1=y_s-A1*x_s;
	//C1=C/(A-x_s);
	//D1=-C1*A;

	D = 1 / (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) + alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU));
	A1 = (alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(un_e - uDetectorCenterPixel_GPU) - (-betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)))*D;
	B1 = ((un_e - uDetectorCenterPixel_GPU)*focusObjectDistance_GPU)*D;
	C1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(alphaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU + A1*betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU);
	D1 = gammaIOcylinderC_GPU*(vn_e - vDetectorCenterPixel_GPU)*(betaIOcylinderC_GPU[phi]*xVolumePixelSize_GPU*B1 + focusObjectDistance_GPU);
	//A1=(B-y_s)/(A-x_s);
	//B1=(B-A1*A)/xVolumePixelSize_GPU;
	//C1=C/(A-x_s);
	//D1=(C-C1*A)/xVolumePixelSize_GPU;

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
				x_s+=xVolumePixelSize_GPU*tex3D(volume_tex,A+0.5f,B+0.5f,C+0.5f); //temp

				//Echantillonnage de delta_xn
				A += ctheta; //x_i
				B += stheta; //y_i
				C += salpha; //z_i
			}
			sinogram_g[adresse_une_vne]=x_s;
		}
		else
			sinogram_g[adresse_une_vne]=0;

	}
	else
		sinogram_g[adresse_une_vne]=0;
}

#endif
