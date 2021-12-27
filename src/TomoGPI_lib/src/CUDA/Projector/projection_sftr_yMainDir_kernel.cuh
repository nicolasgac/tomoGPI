/* kernel for SFTR projector
 * when the source is closer
 * to y-axis
 * Author : Camille Chapdelaine
 * 19/10/2017
 */
 #ifndef _PROJECTION_SFR_YMAINDIR_KERNEL_H_
 #define _PROJECTION_SFR_YMAINDIR_KERNEL_H_
//unweighted diagonal coefficients
template <typename T>
__global__ void CoeffDiagHHT_sftr_yMainDir_kernel(T* sinogram_g)
{

	// volume copié sur la mémoire de texture
	unsigned long int phi,un_e, vn_e;
	unsigned int u_min, u_max;
	float proj;
	float xn_prime,yn_prime,zn_prime,un_prime,vn_prime;
	float ksi,ksi_moins,ksi_plus,xphi,yphi,v_moins,v_plus;
	unsigned int xn_e,yn_e,zn_e,ksie_min,ksie_max,zne_min,zne_max;
	float f_ax, f_tr,u_moins,u_plus,s1,s2;
	float tau0, tau1, tau2, tau3,dxsp,dysp;
	bool in_u;

	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y;//+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	un_prime=((float)un_e-uDetectorCenterPixel_GPU);
	vn_prime=((float)vn_e-vDetectorCenterPixel_GPU);

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alphaIOcylinderC_GPU[phi]-un_prime*uDetectorPixelSize_GPU*betaIOcylinderC_GPU[phi];
	//y_p-y_s
	dysp=focusDetectorDistance_GPU*betaIOcylinderC_GPU[phi]+un_prime*uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi];

	// the source is closer to y-axis
	if((phi>=projectionNb_GPU/8 && phi<3*projectionNb_GPU/8)||(phi>=5*projectionNb_GPU/8 && phi<7*projectionNb_GPU/8)){

		// initialize
		proj=0;

		// y-axis is the primary axis
		for(yn_e=0;yn_e<yVolumePixelNb_GPU;yn_e++){
			yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;
			ksi=-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*alphaIOcylinderC_GPU[phi]+((dxsp/dysp)*(yn_prime+(focusObjectDistance_GPU/xVolumePixelSize_GPU)*betaIOcylinderC_GPU[phi]));
			ksi=ksi+xVolumeCenterPixel_GPU;

			// neighbours in x-axis
			ksi_moins=roundf(ksi)-1.0;//ksi-0.5;
			if(ksi_moins<0){
				ksi_moins=0;
			}
			ksie_min=(unsigned int) ksi_moins;//floor(ksi_moins);
			ksi_plus=roundf(ksi)+1.0;//ksi+0.5;
			ksie_max=(unsigned int) ksi_plus;//ceil(ksi_plus);
			if(ksie_max>xVolumePixelNb_GPU-1){
				ksie_max=xVolumePixelNb_GPU-1;
			}
			for(xn_e=ksie_min;xn_e<=ksie_max;xn_e++){
				xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
				yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

				// transaxial footprint
				// project the four corners and build the trapezoid
				// corner (-0.5,-0.5) : p1
				v_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (0.5,-0.5) : p2
				v_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (-0.5,0.5) : p3
				u_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				// corner (0.5,0.5) : p4
				u_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				tau0=min(min(v_moins,v_plus),min(u_moins,u_plus));
				tau3=max(max(v_moins,v_plus),max(u_moins,u_plus));
				s1=max(min(v_moins,v_plus),min(u_moins,u_plus));
				s2=min(max(v_moins,v_plus),max(u_moins,u_plus));
				tau1=min(s1,s2);
				tau2=max(s1,s2);
				u_moins=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5
				if(u_moins<0){
					u_moins=0;
				}
				u_min=floor(u_moins);
				u_plus=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5
				u_max=ceil(u_plus);
				if(u_max>uDetectorPixelNb_GPU-1){
					u_max=uDetectorPixelNb_GPU-1;
				}

				// the cell lies in the footprint
				in_u=(un_e<=u_max && un_e>=u_min && u_moins<=float(uDetectorPixelNb_GPU-1) && u_plus>=0);

				s1=max((un_prime-0.5),tau0);
				s2=min((un_prime+0.5),tau1);
				if(s1<s2 && in_u){
					v_moins=0.5*((s2-tau0)*(s2-tau0)-(s1-tau0)*(s1-tau0))/(tau1-tau0);
				}else{
					v_moins=0;
				}
				s1=max((un_prime-0.5),tau1);
				s2=min((un_prime+0.5),tau2);
				if(s1<s2 && in_u){
					v_plus=s2-s1;
				}else{
					v_plus=0;
				}
				s1=max((un_prime-0.5),tau2);
				s2=min((un_prime+0.5),tau3);
				if(s1<s2 && in_u){
					u_moins=0.5*((s1-tau3)*(s1-tau3)-(s2-tau3)*(s2-tau3))/(tau3-tau2);
				}else{
					u_moins=0;
				}
				// transaxial footprint
				f_tr=v_moins+v_plus+u_moins;

				// calcul de z pour lesquels F2 non nul
				u_moins=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5
				u_plus=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5
				if(u_moins<0){
					u_moins=0;
				}
				zne_min=floor(u_moins);
				zne_max=ceil(u_plus);
				if(zne_max>zVolumePixelNb_GPU-1){
					zne_max=zVolumePixelNb_GPU-1;
				}

				// axial footprint
				f_ax=0;
				for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
					zn_prime=((float)zn_e-zVolumeCenterPixel_GPU);
					v_moins=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime-0.5)/xphi);
					v_plus=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime+0.5)/xphi);

					s1=max((vn_prime-0.5),v_moins);
					s2=min((vn_prime+0.5),v_plus);
					if(s1<s2 && u_moins<=float(zVolumePixelNb_GPU-1) && u_plus>=0 && ksi_moins<=float(xVolumePixelNb_GPU-1) && ksi_plus>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU))){
						f_ax+=(s2-s1)*(s2-s1);
					}
				}

				//A2 method
				s1=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
				s2=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
				f_ax=f_ax*f_tr*f_tr;
				proj+=f_ax*((xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU*xVolumePixelSize_GPU/(max(abs(s1),abs(s2))*max(abs(s1),abs(s2))));

			}
		}


		// méthode A2
		un_prime=un_prime*uDetectorPixelSize_GPU;
		vn_prime=vn_prime*vDetectorPixelSize_GPU;
		sinogram_g[un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU]=proj*(1+((vn_prime*vn_prime)/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU)));

	}

}

// weighted diagonal coefficients
template <typename T>
__global__ void weightedCoeffDiagHVHT_sftr_yMainDir_kernel(T* sinogram_g)
{

	// volume copié sur la mémoire de texture
	unsigned long int phi,un_e, vn_e;
	unsigned int u_min, u_max;
	float proj;
	float xn_prime,yn_prime,zn_prime,un_prime,vn_prime;
	float ksi,ksi_moins,ksi_plus,xphi,yphi,v_moins,v_plus;
	unsigned int xn_e,yn_e,zn_e,ksie_min,ksie_max,zne_min,zne_max;
	float f_ax, f_tr,u_moins,u_plus,s1,s2;
	float tau0, tau1, tau2, tau3,dxsp,dysp;
	bool in_u;

	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y;//+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	un_prime=((float)un_e-uDetectorCenterPixel_GPU);
	vn_prime=((float)vn_e-vDetectorCenterPixel_GPU);

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alphaIOcylinderC_GPU[phi]-un_prime*uDetectorPixelSize_GPU*betaIOcylinderC_GPU[phi];
	//y_p-y_s
	dysp=focusDetectorDistance_GPU*betaIOcylinderC_GPU[phi]+un_prime*uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi];

	// the source is closer to y-axis
	if((phi>=projectionNb_GPU/8 && phi<3*projectionNb_GPU/8)||(phi>=5*projectionNb_GPU/8 && phi<7*projectionNb_GPU/8)){

		// initialize
		proj=0;

		// y-axis is the primary axis
		for(yn_e=0;yn_e<yVolumePixelNb_GPU;yn_e++){
			yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;
			ksi=-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*alphaIOcylinderC_GPU[phi]+((dxsp/dysp)*(yn_prime+(focusObjectDistance_GPU/xVolumePixelSize_GPU)*betaIOcylinderC_GPU[phi]));
			ksi=ksi+xVolumeCenterPixel_GPU;

			// neighbours in x-axis
			ksi_moins=roundf(ksi)-1.0;//ksi-0.5;
			if(ksi_moins<0){
				ksi_moins=0;
			}
			ksie_min=(unsigned int) ksi_moins;//floor(ksi_moins);
			ksi_plus=roundf(ksi)+1.0;//ksi+0.5;
			ksie_max=(unsigned int) ksi_plus;//ceil(ksi_plus);
			if(ksie_max>xVolumePixelNb_GPU-1){
				ksie_max=xVolumePixelNb_GPU-1;
			}
			for(xn_e=ksie_min;xn_e<=ksie_max;xn_e++){
				xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
				yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

				// transaxial footprint
				// project the four corners and build the trapezoid
				// corner (-0.5,-0.5) : p1
				v_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (0.5,-0.5) : p2
				v_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (-0.5,0.5) : p3
				u_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				// corner (0.5,0.5) : p4
				u_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				tau0=min(min(v_moins,v_plus),min(u_moins,u_plus));
				tau3=max(max(v_moins,v_plus),max(u_moins,u_plus));
				s1=max(min(v_moins,v_plus),min(u_moins,u_plus));
				s2=min(max(v_moins,v_plus),max(u_moins,u_plus));
				tau1=min(s1,s2);
				tau2=max(s1,s2);
				u_moins=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5
				if(u_moins<0){
					u_moins=0;
				}
				u_min=floor(u_moins);
				u_plus=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5
				u_max=ceil(u_plus);
				if(u_max>uDetectorPixelNb_GPU-1){
					u_max=uDetectorPixelNb_GPU-1;
				}

				// the cell lies in the footprint
				in_u=(un_e<=u_max && un_e>=u_min && u_moins<=float(uDetectorPixelNb_GPU-1) && u_plus>=0);

				s1=max((un_prime-0.5),tau0);
				s2=min((un_prime+0.5),tau1);
				if(s1<s2 && in_u){
					v_moins=0.5*((s2-tau0)*(s2-tau0)-(s1-tau0)*(s1-tau0))/(tau1-tau0);
				}else{
					v_moins=0;
				}
				s1=max((un_prime-0.5),tau1);
				s2=min((un_prime+0.5),tau2);
				if(s1<s2 && in_u){
					v_plus=s2-s1;
				}else{
					v_plus=0;
				}
				s1=max((un_prime-0.5),tau2);
				s2=min((un_prime+0.5),tau3);
				if(s1<s2 && in_u){
					u_moins=0.5*((s1-tau3)*(s1-tau3)-(s2-tau3)*(s2-tau3))/(tau3-tau2);
				}else{
					u_moins=0;
				}
				// transaxial footprint
				f_tr=v_moins+v_plus+u_moins;

				// calcul de z pour lesquels F2 non nul
				u_moins=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5
				u_plus=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5
				if(u_moins<0){
					u_moins=0;
				}
				zne_min=floor(u_moins);
				zne_max=ceil(u_plus);
				if(zne_max>zVolumePixelNb_GPU-1){
					zne_max=zVolumePixelNb_GPU-1;
				}

				// axial footprint
				f_ax=0;
				for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
					zn_prime=((float)zn_e-zVolumeCenterPixel_GPU);
					v_moins=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime-0.5)/xphi);
					v_plus=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime+0.5)/xphi);

					s1=max((vn_prime-0.5),v_moins);
					s2=min((vn_prime+0.5),v_plus);
					if(s1<s2 && u_moins<=float(zVolumePixelNb_GPU-1) && u_plus>=0 && ksi_moins<=float(xVolumePixelNb_GPU-1) && ksi_plus>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU))){
						f_ax+=(s2-s1)*(s2-s1)*tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);//d_volume[adresse_xne_yne_zne]
					}
				}

				//A2 method
				s1=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
				s2=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
				f_ax=f_ax*f_tr*f_tr;
				proj+=f_ax*((xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU*xVolumePixelSize_GPU/(max(abs(s1),abs(s2))*max(abs(s1),abs(s2))));

			}
		}


		// méthode A2
		un_prime=un_prime*uDetectorPixelSize_GPU;
		vn_prime=vn_prime*vDetectorPixelSize_GPU;
		sinogram_g[un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU]=proj*(1+((vn_prime*vn_prime)/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU)));

	}

}

// projector
template <typename T>
__global__ void projection_sftr_yMainDir_kernel(T* sinogram_g)
{

	// volume copié sur la mémoire de texture
	unsigned long int phi,un_e, vn_e;
	unsigned int u_min, u_max;
	float proj;
	float xn_prime,yn_prime,zn_prime,un_prime,vn_prime;
	float ksi,ksi_moins,ksi_plus,xphi,yphi,v_moins,v_plus;
	unsigned int xn_e,yn_e,zn_e,ksie_min,ksie_max,zne_min,zne_max;
	float f_ax, f_tr,u_moins,u_plus,s1,s2;
	float tau0, tau1, tau2, tau3,dxsp,dysp;
	bool in_u;

	un_e=threadIdx.x+blockIdx.x*blockDim.x;
	vn_e=threadIdx.y+blockIdx.y*blockDim.y;//+vn_start;
	phi=threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	un_prime=((float)un_e-uDetectorCenterPixel_GPU);
	vn_prime=((float)vn_e-vDetectorCenterPixel_GPU);

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alphaIOcylinderC_GPU[phi]-un_prime*uDetectorPixelSize_GPU*betaIOcylinderC_GPU[phi];
	//y_p-y_s
	dysp=focusDetectorDistance_GPU*betaIOcylinderC_GPU[phi]+un_prime*uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi];

	// the source is closer to y-axis
	if((phi>=projectionNb_GPU/8 && phi<3*projectionNb_GPU/8)||(phi>=5*projectionNb_GPU/8 && phi<7*projectionNb_GPU/8)){

		// initialize
		proj=0;

		// y-axis is the primary axis
		for(yn_e=0;yn_e<yVolumePixelNb_GPU;yn_e++){
			yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;
			ksi=-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*alphaIOcylinderC_GPU[phi]+((dxsp/dysp)*(yn_prime+(focusObjectDistance_GPU/xVolumePixelSize_GPU)*betaIOcylinderC_GPU[phi]));
			ksi=ksi+xVolumeCenterPixel_GPU;

			// neighbours in x-axis
			ksi_moins=roundf(ksi)-1.0;//ksi-0.5;
			if(ksi_moins<0){
				ksi_moins=0;
			}
			ksie_min=(unsigned int) ksi_moins;//floor(ksi_moins);
			ksi_plus=roundf(ksi)+1.0;//ksi+0.5;
			ksie_max=(unsigned int) ksi_plus;//ceil(ksi_plus);
			if(ksie_max>xVolumePixelNb_GPU-1){
				ksie_max=xVolumePixelNb_GPU-1;
			}
			for(xn_e=ksie_min;xn_e<=ksie_max;xn_e++){
				xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
				yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

				// transaxial footprint
				// project the four corners and build the trapezoid
				// corner (-0.5,-0.5) : p1
				v_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (0.5,-0.5) : p2
				v_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
				// corner (-0.5,0.5) : p3
				u_moins=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				// corner (0.5,0.5) : p4
				u_plus=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
				tau0=min(min(v_moins,v_plus),min(u_moins,u_plus));
				tau3=max(max(v_moins,v_plus),max(u_moins,u_plus));
				s1=max(min(v_moins,v_plus),min(u_moins,u_plus));
				s2=min(max(v_moins,v_plus),max(u_moins,u_plus));
				tau1=min(s1,s2);
				tau2=max(s1,s2);
				u_moins=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5
				if(u_moins<0){
					u_moins=0;
				}
				u_min=floor(u_moins);
				u_plus=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5
				u_max=ceil(u_plus);
				if(u_max>uDetectorPixelNb_GPU-1){
					u_max=uDetectorPixelNb_GPU-1;
				}

				// the cell lies in the footprint
				in_u=(un_e<=u_max && un_e>=u_min && u_moins<=float(uDetectorPixelNb_GPU-1) && u_plus>=0);

				s1=max((un_prime-0.5),tau0);
				s2=min((un_prime+0.5),tau1);
				if(s1<s2 && in_u){
					v_moins=0.5*((s2-tau0)*(s2-tau0)-(s1-tau0)*(s1-tau0))/(tau1-tau0);
				}else{
					v_moins=0;
				}
				s1=max((un_prime-0.5),tau1);
				s2=min((un_prime+0.5),tau2);
				if(s1<s2 && in_u){
					v_plus=s2-s1;
				}else{
					v_plus=0;
				}
				s1=max((un_prime-0.5),tau2);
				s2=min((un_prime+0.5),tau3);
				if(s1<s2 && in_u){
					u_moins=0.5*((s1-tau3)*(s1-tau3)-(s2-tau3)*(s2-tau3))/(tau3-tau2);
				}else{
					u_moins=0;
				}
				// transaxial footprint
				f_tr=v_moins+v_plus+u_moins;

				// calcul de z pour lesquels F2 non nul
				u_moins=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5
				u_plus=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5
				if(u_moins<0){
					u_moins=0;
				}
				zne_min=floor(u_moins);
				zne_max=ceil(u_plus);
				if(zne_max>zVolumePixelNb_GPU-1){
					zne_max=zVolumePixelNb_GPU-1;
				}

				// axial footprint
				f_ax=0;
				for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
					zn_prime=((float)zn_e-zVolumeCenterPixel_GPU);
					v_moins=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime-0.5)/xphi);
					v_plus=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((zn_prime+0.5)/xphi);

					s1=max((vn_prime-0.5),v_moins);
					s2=min((vn_prime+0.5),v_plus);
					if(s1<s2 && u_moins<=float(zVolumePixelNb_GPU-1) && u_plus>=0 && ksi_moins<=float(xVolumePixelNb_GPU-1) && ksi_plus>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU))){
						f_ax+=(s2-s1)*tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);//d_volume[adresse_xne_yne_zne]
					}
				}

				//A2 method
				s1=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
				s2=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
				f_ax=f_ax*f_tr;
				proj+=f_ax*(sqrtf(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(s1),abs(s2))));

			}
		}


		// méthode A2
		un_prime=un_prime*uDetectorPixelSize_GPU;
		vn_prime=vn_prime*vDetectorPixelSize_GPU;
		sinogram_g[un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU]=proj*sqrtf(1+((vn_prime*vn_prime)/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU)));

	}

}

#endif
