/* merge kernels of SFTR projector
*
* Author : Camille Chapdelaine
* August 03, 2018
*/
#ifndef _PROJECTION_SFR_MERGEDDIR_KERNEL_H_
#define _PROJECTION_SFR_MERGEDDIR_KERNEL_H_



//Première Version (Original)
// projector using shared memory
template <typename T>
__global__ void projection_sftr_generalized_kernel_v1(T* sinogram_g)
{
	// volume copié sur la mémoire de texture
	unsigned long int phi,un_e, vn_e;
	float proj;
	float xn_prime,yn_prime,un_prime,vn_prime;
	float xphi,yphi, dxsp,dysp;
	unsigned int xn_e,yn_e,zn_e,zne_min,zne_max;
	float x_e,y_e;
	float tau0, tau1, tau2, tau3, f_tr0, f_ax0;
	bool in_u;
	float pas_x, pas_y;
	unsigned int ksi;
	unsigned long int ksi_max;

	// transaxial footprint in shared memory
	__shared__ float f_tr[1024];

	un_e = threadIdx.x+blockIdx.x*blockDim.x;
	vn_e = threadIdx.y+blockIdx.y*blockDim.y;//+vn_start;
	phi = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	//adress=un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;
	//printf("%llu",adress);
	un_prime=((float)un_e-uDetectorCenterPixel_GPU);
	vn_prime=((float)vn_e-vDetectorCenterPixel_GPU);

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alphaIOcylinderC_GPU[phi]-un_prime*uDetectorPixelSize_GPU*betaIOcylinderC_GPU[phi];
	//y_p-y_s
	dysp=focusDetectorDistance_GPU*betaIOcylinderC_GPU[phi]+un_prime*uDetectorPixelSize_GPU*alphaIOcylinderC_GPU[phi];

	// initialize x and y indices
	if(abs(dxsp)>=abs(dysp))
	{
		pas_x=1.0;
		pas_y=dysp/dxsp;
		ksi_max=xVolumePixelNb_GPU;
		x_e=0.0;
		y_e=yVolumeCenterPixel_GPU-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*betaIOcylinderC_GPU[phi]-pas_y*(xVolumeCenterPixel_GPU-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*alphaIOcylinderC_GPU[phi]);
	}else
	{
		pas_x=dxsp/dysp;
		pas_y=1.0;
		ksi_max=yVolumePixelNb_GPU;
		y_e=0.0;
		x_e=xVolumeCenterPixel_GPU-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*alphaIOcylinderC_GPU[phi]-pas_x*(yVolumeCenterPixel_GPU-(focusObjectDistance_GPU/xVolumePixelSize_GPU)*betaIOcylinderC_GPU[phi]);
	}

	// initialize projection
	proj=0.0;

	// loop on ksi
	for(ksi=0;ksi<ksi_max;ksi++){

		/*******************
		*
		* first duplet
		*
		* *****************/
		xphi=roundf(x_e)-float(ceil(1-abs(pas_x)));
		yphi=roundf(y_e)-float(ceil(1-abs(pas_y)));
		if(xphi<0){
			xn_e=0;
		}else if(xphi>xVolumePixelNb_GPU-1){
			xn_e=xVolumePixelNb_GPU-1;
		}else{
			xn_e=(unsigned int) xphi;
		}
		if(yphi<0){
			yn_e=0;
		}else if(yphi>yVolumePixelNb_GPU-1){
			yn_e=yVolumePixelNb_GPU-1;
		}else{
			yn_e=(unsigned int) yphi;
		}
		// compute footprint
		xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
		yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;
		// in the field-of-view
		#if FOV_CYLINDER
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU)));
		#else
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0);
		#endif
		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
		yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		xn_prime=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5 : z_plus
		if(xn_prime<0){
			xn_prime=0;
		}
		zne_min=floor(xn_prime);
		zne_max=ceil(yn_prime);
		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime<=float(zVolumePixelNb_GPU-1) && yn_prime>=0  && in_u);

		// axial footprint
		f_ax0=0;
		xn_prime=((float)zne_min-zVolumeCenterPixel_GPU);
		yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime-0.5)/xphi);//-1
		for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
			dxsp=max((vn_prime-0.5),yn_prime);//s1
			yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime+0.5)/xphi);//+1
			dysp=min((vn_prime+0.5),yn_prime);//s2
			if(dxsp<dysp && in_u && (abs(pas_x)<1.0 || abs(pas_y)<1.0)){
				f_ax0+=(dysp-dxsp)*tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
			}
			xn_prime=((float)(zn_e+1)-zVolumeCenterPixel_GPU);
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			// corner (-0.5,-0.5) : p1
			dxsp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (0.5,-0.5) : p2
			dysp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (-0.5,0.5) : p3
			xn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			// corner (0.5,0.5) : p4
			yn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			tau0=max(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			if(xn_prime<0){
				xn_prime=0;
			}
			zne_min=floor(xn_prime);// u_min
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			zne_max=ceil(yn_prime);// u_max
			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=zne_min && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			// transaxial footprint
			f_tr0=0.0;
			xn_prime=max((un_prime-0.5),tau0);//s1
			yn_prime=min((un_prime+0.5),tau1);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((yn_prime-tau0)*(yn_prime-tau0)-(xn_prime-tau0)*(xn_prime-tau0))/(tau1-tau0));
			}
			xn_prime=max((un_prime-0.5),tau1);//s1
			yn_prime=min((un_prime+0.5),tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max((un_prime-0.5),tau2);//s1
			yn_prime=min((un_prime+0.5),tau3);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((xn_prime-tau3)*(xn_prime-tau3)-(yn_prime-tau3)*(yn_prime-tau3))/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
			yn_prime=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
			f_tr0=f_tr0*(sqrtf(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			f_tr[threadIdx.x]=f_tr0;
		}
		__syncthreads();

		//update projections
		proj+=(f_tr[threadIdx.x]*f_ax0);

		/*******************
		*
		* second duplet
		*
		* *****************/
		xphi=roundf(x_e);
		yphi=roundf(y_e);
		if(xphi<0){
			xn_e=0;
		}else if(xphi>xVolumePixelNb_GPU-1){
			xn_e=xVolumePixelNb_GPU-1;
		}else{
			xn_e=(unsigned int) xphi;
		}
		if(yphi<0){
			yn_e=0;
		}else if(yphi>yVolumePixelNb_GPU-1){
			yn_e=yVolumePixelNb_GPU-1;
		}else{
			yn_e=(unsigned int) yphi;
		}

		// compute footprint
		xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
		yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;

		// in the field-of-view
		#if FOV_CYLINDER
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU)));
		#else
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0);
		#endif
		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
		yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		xn_prime=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5 : z_plus
		if(xn_prime<0){
			xn_prime=0;
		}
		zne_min=floor(xn_prime);
		zne_max=ceil(yn_prime);
		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime<=float(zVolumePixelNb_GPU-1) && yn_prime>=0  && in_u);

		// axial footprint
		f_ax0=0;
		xn_prime=((float)zne_min-zVolumeCenterPixel_GPU);
		yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime-0.5)/xphi);//-1
		for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
			dxsp=max((vn_prime-0.5),yn_prime);//s1
			yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime+0.5)/xphi);//+1
			dysp=min((vn_prime+0.5),yn_prime);//s2
			if(dxsp<dysp && in_u){
				f_ax0+=(dysp-dxsp)*tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
			}
			xn_prime=((float)(zn_e+1)-zVolumeCenterPixel_GPU);
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			// corner (-0.5,-0.5) : p1
			dxsp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (0.5,-0.5) : p2
			dysp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (-0.5,0.5) : p3
			xn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			// corner (0.5,0.5) : p4
			yn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			tau0=max(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			if(xn_prime<0){
				xn_prime=0;
			}
			zne_min=floor(xn_prime);// u_min
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			zne_max=ceil(yn_prime);// u_max
			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=zne_min && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			// transaxial footprint
			f_tr0=0.0;
			xn_prime=max((un_prime-0.5),tau0);//s1
			yn_prime=min((un_prime+0.5),tau1);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((yn_prime-tau0)*(yn_prime-tau0)-(xn_prime-tau0)*(xn_prime-tau0))/(tau1-tau0));
			}
			xn_prime=max((un_prime-0.5),tau1);//s1
			yn_prime=min((un_prime+0.5),tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max((un_prime-0.5),tau2);//s1
			yn_prime=min((un_prime+0.5),tau3);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((xn_prime-tau3)*(xn_prime-tau3)-(yn_prime-tau3)*(yn_prime-tau3))/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
			yn_prime=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
			f_tr0=f_tr0*(sqrtf(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			f_tr[threadIdx.x]=f_tr0;
		}
		__syncthreads();

		//update projections
		proj+=(f_tr[threadIdx.x]*f_ax0);

		/*******************
		*
		* third and last duplet
		*
		* *****************/
		xphi=roundf(x_e)+ceilf(1-abs(pas_x));
		yphi=roundf(y_e)+ceilf(1-abs(pas_y));
		if(xphi<0){
			xn_e=0;
		}else if(xphi>xVolumePixelNb_GPU-1){
			xn_e=xVolumePixelNb_GPU-1;
		}else{
			xn_e=(unsigned int) xphi;
		}
		if(yphi<0){
			yn_e=0;
		}else if(yphi>yVolumePixelNb_GPU-1){
			yn_e=yVolumePixelNb_GPU-1;
		}else{
			yn_e=(unsigned int) yphi;
		}
		// compute footprint
		xn_prime=(float)xn_e-xVolumeCenterPixel_GPU;
		yn_prime=(float)yn_e-yVolumeCenterPixel_GPU;
		// in the field-of-view
		#if FOV_CYLINDER
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0 && ((xn_prime*xn_prime+yn_prime*yn_prime)<0.25*float(xVolumePixelNb_GPU)*float(xVolumePixelNb_GPU)));
		#else
		in_u=(xphi<=float(xVolumePixelNb_GPU-1) && xphi>=0 && yphi<=float(yVolumePixelNb_GPU-1) && yphi>=0);
		#endif
		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alphaIOcylinderC_GPU[phi]*xn_prime+betaIOcylinderC_GPU[phi]*yn_prime;
		yphi=-betaIOcylinderC_GPU[phi]*xn_prime+alphaIOcylinderC_GPU[phi]*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		xn_prime=zVolumeCenterPixel_GPU-0.5+((vn_prime-0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+((vn_prime+0.5)*xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU);//+0.5 : z_plus
		if(xn_prime<0){
			xn_prime=0;
		}
		zne_min=floor(xn_prime);
		zne_max=ceil(yn_prime);
		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime<=float(zVolumePixelNb_GPU-1) && yn_prime>=0  && in_u);

		// axial footprint
		f_ax0=0;
		xn_prime=((float)zne_min-zVolumeCenterPixel_GPU);
		yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime-0.5)/xphi);//-1
		for(zn_e=zne_min;zn_e<=zne_max;zn_e++){
			dxsp=max((vn_prime-0.5),yn_prime);//s1
			yn_prime=(focusDetectorDistance_GPU/vDetectorPixelSize_GPU)*((xn_prime+0.5)/xphi);//+1
			dysp=min((vn_prime+0.5),yn_prime);//s2
			if(dxsp<dysp && in_u && (abs(pas_x)<1.0 || abs(pas_y)<1.0)){
				f_ax0+=(dysp-dxsp)*tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
			}
			xn_prime=((float)(zn_e+1)-zVolumeCenterPixel_GPU);
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			// corner (-0.5,-0.5) : p1
			dxsp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (0.5,-0.5) : p2
			dysp=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]-alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]-betaIOcylinderC_GPU[phi]));
			// corner (-0.5,0.5) : p3
			xn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(-alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			// corner (0.5,0.5) : p4
			yn_prime=(focusDetectorDistance_GPU/uDetectorPixelSize_GPU)*(yphi+0.5*(-betaIOcylinderC_GPU[phi]+alphaIOcylinderC_GPU[phi]))/(xphi+0.5*(alphaIOcylinderC_GPU[phi]+betaIOcylinderC_GPU[phi]));
			tau0=max(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),min(xn_prime,yn_prime));
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			if(xn_prime<0){
				xn_prime=0;
			}
			zne_min=floor(xn_prime);// u_min
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			zne_max=ceil(yn_prime);// u_max
			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=zne_min && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			// transaxial footprint
			f_tr0=0.0;
			xn_prime=max((un_prime-0.5),tau0);//s1
			yn_prime=min((un_prime+0.5),tau1);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((yn_prime-tau0)*(yn_prime-tau0)-(xn_prime-tau0)*(xn_prime-tau0))/(tau1-tau0));
			}
			xn_prime=max((un_prime-0.5),tau1);//s1
			yn_prime=min((un_prime+0.5),tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max((un_prime-0.5),tau2);//s1
			yn_prime=min((un_prime+0.5),tau3);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*((xn_prime-tau3)*(xn_prime-tau3)-(yn_prime-tau3)*(yn_prime-tau3))/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alphaIOcylinderC_GPU[phi]-yphi*betaIOcylinderC_GPU[phi];//cos_psi
			yn_prime=xphi*betaIOcylinderC_GPU[phi]+yphi*alphaIOcylinderC_GPU[phi];//sin_psi
			f_tr0=f_tr0*(sqrtf(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			f_tr[threadIdx.x]=f_tr0;
		}
		__syncthreads();

		//update projections
		proj+=(f_tr[threadIdx.x]*f_ax0);

		/*******************************
		* steps in x- and y-directions
		* *****************************/
		x_e=x_e+pas_x;
		y_e=y_e+pas_y;
	}

	// méthode A2
	un_prime=un_prime*uDetectorPixelSize_GPU;
	vn_prime=vn_prime*vDetectorPixelSize_GPU;
	sinogram_g[un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU]=proj*sqrtf(1+((vn_prime*vn_prime)/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU)));


}


//Nouvelle implémentation Projector v2
// projector using shared memory
template <typename T>
//__global__ void projection_sftr_generalized_kernel_v2(int K , int BLOCK_SIZE, T* sinogram_g)
__global__ void projection_sftr_generalized_kernel_v2(int K, T* sinogram_g)
{
	// volume copié sur la mémoire de texture

	unsigned int un_e, vn_e, v_min,v_max,zne_max, BLOCK_SIZE;
	float xn_prime,yn_prime,un_prime,vn_prime;
	float xphi,yphi,dxsp,dysp;
	short int xn_e,yn_e,zn_e,id;
	float tau0, tau1, tau2, tau3, f_tr0;
	float pas_x, pas_y;
	short int ksi;
	short int ksi_max;
	float alpha, beta;
	float x_e, y_e;
	float temp1 = vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
	float temp2 = focusDetectorDistance_GPU/uDetectorPixelSize_GPU;


	un_e = threadIdx.x+blockIdx.x*blockDim.x;
	vn_e = (threadIdx.y+blockIdx.y*blockDim.y)*K;
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;


	un_prime=float(un_e)-uDetectorCenterPixel_GPU;
	vn_prime=float(vn_e)-vDetectorCenterPixel_GPU;

	v_min = vn_e;

	id = threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;


	BLOCK_SIZE = blockDim.x*blockDim.y*blockDim.z;


	// transaxial footprint in shared memory
	extern __shared__ float s[];
	float *f_ax0=s;
	float *proj = (float*)&f_ax0[K*BLOCK_SIZE];


	alpha = alphaIOcylinderC_GPU[zne_max];
	beta = betaIOcylinderC_GPU[zne_max];

	dysp  = un_prime*uDetectorPixelSize_GPU;
	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alpha-dysp*beta;

	//y_p-y_s
	dysp=focusDetectorDistance_GPU*beta+dysp*alpha;

	// initialize x and y indices
	tau0 = (focusObjectDistance_GPU/xVolumePixelSize_GPU);
	if(abs(dxsp)>=abs(dysp))
	{
		pas_x=1.0;
		pas_y=dysp/dxsp;
		ksi_max=xVolumePixelNb_GPU;
		x_e=(0.0);
		y_e=(yVolumeCenterPixel_GPU-tau0*beta-pas_y*(xVolumeCenterPixel_GPU-tau0*alpha));
	}
	else
	{
		pas_x=dxsp/dysp;
		pas_y=1.0;
		ksi_max=yVolumePixelNb_GPU;
		y_e=(0.0);
		x_e=(xVolumeCenterPixel_GPU-tau0*alpha-pas_x*(yVolumeCenterPixel_GPU-tau0*beta));
	}

	// initialize projection
	for(vn_e = 0; vn_e<K; vn_e++){
		proj[id+vn_e*BLOCK_SIZE] = 0;
		f_ax0[id+vn_e*BLOCK_SIZE] = 0;
	}

	// loop on ksi
	for(ksi=0;ksi<ksi_max;ksi++){

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* first duplet
		*
		* *****************/
		if (abs(pas_x)< 1.0f || abs(pas_y) <1.0f){
			xphi=roundf(x_e)-ceilf(1.0f-abs(pas_x));
			yphi=roundf(y_e)-ceilf(1.0f-abs(pas_y));


			if (xphi < 0.){
				xn_e = 0;
			}
			else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
				xn_e = (unsigned int) xVolumePixelNb_GPU-1;
			}
			else{
				xn_e = (unsigned int) xphi;
			}

			if (yphi < 0.){
				yn_e = 0;
			}
			else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
				yn_e = (unsigned int) yVolumePixelNb_GPU-1;
			}
			else{
				yn_e = (unsigned int) yphi;
			}

			// compute footprint
			xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
			yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

			#if FOV_CYLINDER
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
			#else
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
			#endif
			{
				// in the field-of-view
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
				yphi=-beta*xn_prime+alpha*yn_prime;

				// compute axial footprint
				// axial footprint : calcul de z pour lesquels F2 non nul
				tau1 = vn_prime-0.5f;
				tau2 = xphi*temp1;
				xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
				yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

				if(xn_prime<0.){
					xn_prime=0.;
				}

				v_max=floor(xn_prime);//z_min
				zne_max=ceil(yn_prime);

				if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
					zne_max= (unsigned int) zVolumePixelNb_GPU-1;
				}

				//axial footprint

				if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
					f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
					tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
					xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
					tau0 = tau3*xn_prime;//chi0
					tau1 = tau3*(xn_prime+1);//chi1
					tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
					yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
					for(zn_e=v_max;zn_e<=zne_max;zn_e++){

						//bornes
						vn_e = floor(tau2);
						if (vn_e < v_min){
							vn_e = v_min;
						}
						vn_e -= v_min;
						v_max = ceil(yn_prime)-v_min;

						if (v_max > K){
							v_max = K;
						}
						vn_prime = float(vn_e) + f_tr0;
						xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+0.5f);
						dxsp = vn_prime+1.0f;
						for (;vn_e < v_max; vn_e++){
							dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
							if(dysp > 0){
								f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
							}
							vn_prime += 1.0f;
							dxsp += 1.0f;
						}
						yn_prime += tau3;
						tau2 += tau3;
						tau0 += tau3;
						tau1 += tau3;
					}


					//compute transaxial footprint

					// projections
					// project the four corners and build the trapezoid
					/*
					// corner (-0.5,-0.5) : p1
					dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
					// corner (0.5,-0.5) : p2
					dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
					// corner (-0.5,0.5) : p3
					xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
					// corner (0.5,0.5) : p4
					yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
					*/

					tau1 = xphi + xphi;
					tau2 = yphi + yphi;

					tau3 = alpha + beta;
					vn_prime = alpha - beta;
					// corner (-0.5,-0.5) : p1
					dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
					// corner (0.5,-0.5) : p2
					dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
					// corner (-0.5,0.5) : p3
					xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
					// corner (0.5,0.5) : p4
					yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


					f_tr0 = min(dxsp,dysp);
					vn_prime = min(xn_prime,yn_prime);
					tau0=max(f_tr0,vn_prime);
					tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
					tau1=min(tau0,tau3);
					tau2=max(tau0,tau3);
					tau0=min(f_tr0,vn_prime);
					tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
					xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
					if(xn_prime<0){
						xn_prime=0;
					}
					v_max=floor(xn_prime);// u_min
					yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
					zne_max=ceil(yn_prime);// u_max
					if(zne_max>uDetectorPixelNb_GPU-1){
						zne_max=uDetectorPixelNb_GPU-1;
					}

					// the cell lies in the footprint
					f_tr0=0.0f;
					// transaxial footprint
					if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
						dxsp = un_prime-0.5f;
						dysp = un_prime+0.5f;

						xn_prime=max(dxsp,tau0)-tau0;//s1
						yn_prime=min(dysp,tau1)-tau0;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
						}
						xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
						if(xn_prime> 0){
							f_tr0+=xn_prime;
						}
						xn_prime=max(dxsp,tau2)-tau3;//s1
						yn_prime=min(dysp,tau3)-tau3;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
						}
						//A2 method
						f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

						for (vn_e = 0;vn_e < K; vn_e++){
							proj[id]+=f_tr0*f_ax0[id];
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

					}

					if (f_tr0 == 0){
						for (vn_e = 0;vn_e < K; vn_e++){
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
					}

				}

			}
			//update projections


		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* second duplet
		*
		* *****************/
		xphi=roundf(x_e);
		yphi=roundf(y_e);


		if (xphi < 0.){
			xn_e = 0;
		}
		else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
			xn_e = (unsigned int) xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0.){
			yn_e = 0;
		}
		else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
			yn_e = (unsigned int) yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}

		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		#if FOV_CYLINDER
		if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
		#else
		if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
		#endif
		{
			// in the field-of-view
			xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
			yphi=-beta*xn_prime+alpha*yn_prime;

			// compute axial footprint
			// axial footprint : calcul de z pour lesquels F2 non nul
			tau1 = vn_prime-0.5f;
			tau2 = xphi*temp1;
			xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
			yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

			if(xn_prime<0.){
				xn_prime=0.;
			}

			v_max=floor(xn_prime);//z_min
			zne_max=ceil(yn_prime);

			if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
				zne_max= (unsigned int) zVolumePixelNb_GPU-1;
			}

			//axial footprint

			if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
				f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
				tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
				xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
				tau0 = tau3*xn_prime;//chi0
				tau1 = tau3*(xn_prime+1);//chi1
				tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
				yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
				for(zn_e=v_max;zn_e<=zne_max;zn_e++){

					//bornes
					vn_e = floor(tau2);
					if (vn_e < v_min){
						vn_e = v_min;
					}
					vn_e -= v_min;
					v_max = ceil(yn_prime)-v_min;

					if (v_max > K){
						v_max = K;
					}
					vn_prime = float(vn_e) + f_tr0;
					xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+0.5f);
					dxsp = vn_prime+1.0f;
					for (;vn_e < v_max; vn_e++){
						dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
						if(dysp > 0){
							f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
						}
						vn_prime += 1.0f;
						dxsp += 1.0f;
					}
					yn_prime += tau3;
					tau2 += tau3;
					tau0 += tau3;
					tau1 += tau3;
				}


				//compute transaxial footprint

				// projections
				// project the four corners and build the trapezoid
				/*
				// corner (-0.5,-0.5) : p1
				dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
				// corner (0.5,-0.5) : p2
				dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
				// corner (-0.5,0.5) : p3
				xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
				// corner (0.5,0.5) : p4
				yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
				*/

				tau1 = xphi + xphi;
				tau2 = yphi + yphi;

				tau3 = alpha + beta;
				vn_prime = alpha - beta;
				// corner (-0.5,-0.5) : p1
				dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
				// corner (0.5,-0.5) : p2
				dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
				// corner (-0.5,0.5) : p3
				xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
				// corner (0.5,0.5) : p4
				yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


				f_tr0 = min(dxsp,dysp);
				vn_prime = min(xn_prime,yn_prime);
				tau0=max(f_tr0,vn_prime);
				tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
				tau1=min(tau0,tau3);
				tau2=max(tau0,tau3);
				tau0=min(f_tr0,vn_prime);
				tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
				xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
				if(xn_prime<0){
					xn_prime=0;
				}
				v_max=floor(xn_prime);// u_min
				yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
				zne_max=ceil(yn_prime);// u_max
				if(zne_max>uDetectorPixelNb_GPU-1){
					zne_max=uDetectorPixelNb_GPU-1;
				}

				// the cell lies in the footprint
				f_tr0=0.0f;
				// transaxial footprint
				if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
					dxsp = un_prime-0.5f;
					dysp = un_prime+0.5f;

					xn_prime=max(dxsp,tau0)-tau0;//s1
					yn_prime=min(dysp,tau1)-tau0;//s2
					if(xn_prime<yn_prime){
						f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
					}
					xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
					if(xn_prime> 0){
						f_tr0+=xn_prime;
					}
					xn_prime=max(dxsp,tau2)-tau3;//s1
					yn_prime=min(dysp,tau3)-tau3;//s2
					if(xn_prime<yn_prime){
						f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
					}
					//A2 method
					f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

					for (vn_e = 0;vn_e < K; vn_e++){
						proj[id]+=f_tr0*f_ax0[id];
						f_ax0[id] = 0;
						id += BLOCK_SIZE;
					}
					id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

				}

				if (f_tr0 == 0){
					for (vn_e = 0;vn_e < K; vn_e++){
						f_ax0[id] = 0;
						id += BLOCK_SIZE;
					}
					id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
				}

			}

		}
		//update projections

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* third and last duplet
		*
		* *****************/
		if (abs(pas_x)< 1.0f || abs(pas_y) <1.0f){
			xphi=roundf(x_e)+ceilf(1.0f-abs(pas_x));
			yphi=roundf(y_e)+ceilf(1.0f-abs(pas_y));


			if (xphi < 0.){
				xn_e = 0;
			}
			else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
				xn_e = (unsigned int) xVolumePixelNb_GPU-1;
			}
			else{
				xn_e = (unsigned int) xphi;
			}

			if (yphi < 0.){
				yn_e = 0;
			}
			else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
				yn_e = (unsigned int) yVolumePixelNb_GPU-1;
			}
			else{
				yn_e = (unsigned int) yphi;
			}

			// compute footprint
			xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
			yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

			#if FOV_CYLINDER
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
			#else
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
			#endif
			{
				// in the field-of-view
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
				yphi=-beta*xn_prime+alpha*yn_prime;

				// compute axial footprint
				// axial footprint : calcul de z pour lesquels F2 non nul
				tau1 = vn_prime-0.5f;
				tau2 = xphi*temp1;
				xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
				yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

				if(xn_prime<0.){
					xn_prime=0.;
				}

				v_max=floor(xn_prime);//z_min
				zne_max=ceil(yn_prime);

				if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
					zne_max= (unsigned int) zVolumePixelNb_GPU-1;
				}

				//axial footprint

				if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
					f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
					tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
					xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
					tau0 = tau3*xn_prime;//chi0
					tau1 = tau3*(xn_prime+1);//chi1
					tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
					yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
					for(zn_e=v_max;zn_e<=zne_max;zn_e++){

						//bornes
						vn_e = floor(tau2);
						if (vn_e < v_min){
							vn_e = v_min;
						}
						vn_e -= v_min;
						v_max = ceil(yn_prime)-v_min;

						if (v_max > K){
							v_max = K;
						}
						vn_prime = float(vn_e) + f_tr0;
						xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+0.5f);
						dxsp = vn_prime+1.0f;
						for (;vn_e < v_max; vn_e++){
							dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
							if(dysp > 0){
								f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
							}
							vn_prime += 1.0f;
							dxsp += 1.0f;
						}
						yn_prime += tau3;
						tau2 += tau3;
						tau0 += tau3;
						tau1 += tau3;
					}


					//compute transaxial footprint

					// projections
					// project the four corners and build the trapezoid
					/*
					// corner (-0.5,-0.5) : p1
					dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
					// corner (0.5,-0.5) : p2
					dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
					// corner (-0.5,0.5) : p3
					xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
					// corner (0.5,0.5) : p4
					yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
					*/

					tau1 = xphi + xphi;
					tau2 = yphi + yphi;

					tau3 = alpha + beta;
					vn_prime = alpha - beta;
					// corner (-0.5,-0.5) : p1
					dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
					// corner (0.5,-0.5) : p2
					dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
					// corner (-0.5,0.5) : p3
					xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
					// corner (0.5,0.5) : p4
					yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


					f_tr0 = min(dxsp,dysp);
					vn_prime = min(xn_prime,yn_prime);
					tau0=max(f_tr0,vn_prime);
					tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
					tau1=min(tau0,tau3);
					tau2=max(tau0,tau3);
					tau0=min(f_tr0,vn_prime);
					tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
					xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
					if(xn_prime<0){
						xn_prime=0;
					}
					v_max=floor(xn_prime);// u_min
					yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
					zne_max=ceil(yn_prime);// u_max
					if(zne_max>uDetectorPixelNb_GPU-1){
						zne_max=uDetectorPixelNb_GPU-1;
					}

					// the cell lies in the footprint
					f_tr0=0.0f;
					// transaxial footprint
					if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
						dxsp = un_prime-0.5f;
						dysp = un_prime+0.5f;

						xn_prime=max(dxsp,tau0)-tau0;//s1
						yn_prime=min(dysp,tau1)-tau0;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
						}
						xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
						if(xn_prime> 0){
							f_tr0+=xn_prime;
						}
						xn_prime=max(dxsp,tau2)-tau3;//s1
						yn_prime=min(dysp,tau3)-tau3;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
						}
						//A2 method
						f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

						for (vn_e = 0;vn_e < K; vn_e++){
							proj[id]+=f_tr0*f_ax0[id];
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

					}

					if (f_tr0 == 0){
						for (vn_e = 0;vn_e < K; vn_e++){
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
					}

				}

			}
			//update projections
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************************
		* steps in x- and y-directions
		* *****************************/
		x_e+=pas_x;
		y_e+=pas_y;
	}

	// méthode A2
	un_prime=un_prime*uDetectorPixelSize_GPU;
	f_tr0 = 1/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU);
	vn_prime=(float(v_min)-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	un_e = un_e+(v_min+zne_max*vDetectorPixelNb_GPU)*uDetectorPixelNb_GPU;
	for (v_max = 0; v_max < K; v_max++){
		sinogram_g[un_e]=proj[id + BLOCK_SIZE*v_max]*__fsqrt_rn(1+vn_prime*vn_prime*f_tr0);
		vn_prime += vDetectorPixelSize_GPU;
		un_e += uDetectorPixelNb_GPU;
	}
}


template <typename T>
__global__ void projection_sftr_generalized_kernel_v2_CPU(int K, T* sinogram_g, int vn_start, int zn_start)
{
	// volume copié sur la mémoire de texture
	unsigned int un_e, vn_e, v_min,v_max,zne_max;
	float xn_prime,yn_prime,un_prime,vn_prime;
	float xphi,yphi,dxsp,dysp;
	unsigned int xn_e,yn_e,zn_e,id;
	float tau0, tau1, tau2, tau3, f_tr0;
	float pas_x, pas_y;
	unsigned int ksi;
	unsigned long int ksi_max;
	float alpha, beta;
	float x_e, y_e;

	zn_start = - zn_start;

	float temp1 = vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
	float temp2 = focusDetectorDistance_GPU/uDetectorPixelSize_GPU;

	un_e = threadIdx.x+blockIdx.x*blockDim.x;
	vn_e = (threadIdx.y+blockIdx.y*blockDim.y)*K;//+vn_start; + On traite K-v
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;

	//if (un_e < uDetectorPixelNb_GPU && vn_e + vn_start < vDetectorPixelNb_GPU){

	un_prime=float(un_e)-uDetectorCenterPixel_GPU;
	vn_prime=float(vn_e)-vDetectorCenterPixel_GPU;

	v_min = vn_e+vn_start;

	id = threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;

	unsigned int BLOCK_SIZE = blockDim.x*blockDim.y*blockDim.z;

	extern __shared__ float s[];

	float *f_ax0=s;
	float *proj = (float*)&f_ax0[K*BLOCK_SIZE];


	alpha = alphaIOcylinderC_GPU[zne_max];
	beta = betaIOcylinderC_GPU[zne_max];

	dysp  = un_prime*uDetectorPixelSize_GPU;
	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alpha-dysp*beta;

	//y_p-y_s
	dysp=focusDetectorDistance_GPU*beta+dysp*alpha;

	// initialize x and y indices
	tau0 = (focusObjectDistance_GPU/xVolumePixelSize_GPU);
	if(abs(dxsp)>=abs(dysp))
	{
		pas_x=1.0;
		pas_y=dysp/dxsp;
		ksi_max=xVolumePixelNb_GPU;
		x_e=0.0;
		y_e=yVolumeCenterPixel_GPU-tau0*beta-pas_y*(xVolumeCenterPixel_GPU-tau0*alpha);
	}
	else
	{
		pas_x=dxsp/dysp;
		pas_y=1.0;
		ksi_max=yVolumePixelNb_GPU;
		y_e=0.0;
		x_e=xVolumeCenterPixel_GPU-tau0*alpha-pas_x*(yVolumeCenterPixel_GPU-tau0*beta);
	}

	// initialize projection
	for(vn_e = 0; vn_e<K; vn_e++){
		proj[id+vn_e*BLOCK_SIZE] = 0;
		f_ax0[id+vn_e*BLOCK_SIZE] = 0;
	}

	// loop on ksi
	for(ksi=0;ksi<ksi_max;ksi++){

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* first duplet
		*
		* *****************/
		if (abs(pas_x)< 1.0f || abs(pas_y) <1.0f){
			xphi=roundf(x_e)-ceilf(1.0f-abs(pas_x));
			yphi=roundf(y_e)-ceilf(1.0f-abs(pas_y));


			if (xphi < 0.){
				xn_e = 0;
			}
			else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
				xn_e = (unsigned int) xVolumePixelNb_GPU-1;
			}
			else{
				xn_e = (unsigned int) xphi;
			}

			if (yphi < 0.){
				yn_e = 0;
			}
			else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
				yn_e = (unsigned int) yVolumePixelNb_GPU-1;
			}
			else{
				yn_e = (unsigned int) yphi;
			}

			// compute footprint
			xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
			yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

			#if FOV_CYLINDER
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
			#else
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
			#endif
			{
				// in the field-of-view
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
				yphi=-beta*xn_prime+alpha*yn_prime;

				// compute axial footprint
				// axial footprint : calcul de z pour lesquels F2 non nul
				tau1 = vn_prime-0.5f;
				tau2 = xphi*temp1;
				xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
				yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

				if(xn_prime<0.){
					xn_prime=0.;
				}

				v_max=floor(xn_prime);//z_min
				zne_max=ceil(yn_prime);

				if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
					zne_max= (unsigned int) zVolumePixelNb_GPU-1;
				}

				//axial footprint

				if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
					f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
					tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
					xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
					tau0 = tau3*xn_prime;//chi0
					tau1 = tau3*(xn_prime+1);//chi1
					tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
					yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
					for(zn_e=v_max;zn_e<=zne_max;zn_e++){

						//bornes
						vn_e = floor(tau2);
						if (vn_e < v_min){
							vn_e = v_min;
						}
						vn_e -= v_min;
						v_max = ceil(yn_prime)-v_min;

						if (v_max > K){
							v_max = K;
						}
						vn_prime = float(vn_e) + f_tr0;
						xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+zn_start+0.5f);
						dxsp = vn_prime+1.0f;
						for (;vn_e < v_max; vn_e++){
							dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
							if(dysp > 0){
								f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
							}
							vn_prime += 1.0f;
							dxsp += 1.0f;
						}
						yn_prime += tau3;
						tau2 += tau3;
						tau0 += tau3;
						tau1 += tau3;
					}


					//compute transaxial footprint

					// projections
					// project the four corners and build the trapezoid
					/*
					// corner (-0.5,-0.5) : p1
					dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
					// corner (0.5,-0.5) : p2
					dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
					// corner (-0.5,0.5) : p3
					xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
					// corner (0.5,0.5) : p4
					yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
					*/

					tau1 = xphi + xphi;
					tau2 = yphi + yphi;

					tau3 = alpha + beta;
					vn_prime = alpha - beta;
					// corner (-0.5,-0.5) : p1
					dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
					// corner (0.5,-0.5) : p2
					dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
					// corner (-0.5,0.5) : p3
					xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
					// corner (0.5,0.5) : p4
					yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


					f_tr0 = min(dxsp,dysp);
					vn_prime = min(xn_prime,yn_prime);
					tau0=max(f_tr0,vn_prime);
					tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
					tau1=min(tau0,tau3);
					tau2=max(tau0,tau3);
					tau0=min(f_tr0,vn_prime);
					tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
					xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
					if(xn_prime<0){
						xn_prime=0;
					}
					v_max=floor(xn_prime);// u_min
					yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
					zne_max=ceil(yn_prime);// u_max
					if(zne_max>uDetectorPixelNb_GPU-1){
						zne_max=uDetectorPixelNb_GPU-1;
					}

					// the cell lies in the footprint
					f_tr0=0.0f;
					// transaxial footprint
					if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
						dxsp = un_prime-0.5f;
						dysp = un_prime+0.5f;

						xn_prime=max(dxsp,tau0)-tau0;//s1
						yn_prime=min(dysp,tau1)-tau0;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
						}
						xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
						if(xn_prime> 0){
							f_tr0+=xn_prime;
						}
						xn_prime=max(dxsp,tau2)-tau3;//s1
						yn_prime=min(dysp,tau3)-tau3;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
						}
						//A2 method
						f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

						for (vn_e = 0;vn_e < K; vn_e++){
							proj[id]+=f_tr0*f_ax0[id];
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

					}

					if (f_tr0 == 0){
						for (vn_e = 0;vn_e < K; vn_e++){
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
					}

				}

			}
			//update projections


		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* second duplet
		*
		* *****************/
		xphi=roundf(x_e);
		yphi=roundf(y_e);


		if (xphi < 0.){
			xn_e = 0;
		}
		else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
			xn_e = (unsigned int) xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0.){
			yn_e = 0;
		}
		else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
			yn_e = (unsigned int) yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}

		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		#if FOV_CYLINDER
		if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
		#else
		if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
		#endif
		{
			// in the field-of-view
			xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
			yphi=-beta*xn_prime+alpha*yn_prime;

			// compute axial footprint
			// axial footprint : calcul de z pour lesquels F2 non nul
			tau1 = vn_prime-0.5f;
			tau2 = xphi*temp1;
			xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
			yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

			if(xn_prime<0.){
				xn_prime=0.;
			}

			v_max=floor(xn_prime);//z_min
			zne_max=ceil(yn_prime);

			if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
				zne_max= (unsigned int) zVolumePixelNb_GPU-1;
			}

			//axial footprint

			if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
				f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
				tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
				xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
				tau0 = tau3*xn_prime;//chi0
				tau1 = tau3*(xn_prime+1);//chi1
				tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
				yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
				for(zn_e=v_max;zn_e<=zne_max;zn_e++){

					//bornes
					vn_e = floor(tau2);
					if (vn_e < v_min){
						vn_e = v_min;
					}
					vn_e -= v_min;
					v_max = ceil(yn_prime)-v_min;

					if (v_max > K){
						v_max = K;
					}
					vn_prime = float(vn_e) + f_tr0;
					xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+zn_start+0.5f);
					dxsp = vn_prime+1.0f;
					for (;vn_e < v_max; vn_e++){
						dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
						if(dysp > 0){
							f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
						}
						vn_prime += 1.0f;
						dxsp += 1.0f;
					}
					yn_prime += tau3;
					tau2 += tau3;
					tau0 += tau3;
					tau1 += tau3;
				}


				//compute transaxial footprint

				// projections
				// project the four corners and build the trapezoid
				/*
				// corner (-0.5,-0.5) : p1
				dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
				// corner (0.5,-0.5) : p2
				dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
				// corner (-0.5,0.5) : p3
				xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
				// corner (0.5,0.5) : p4
				yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
				*/

				tau1 = xphi + xphi;
				tau2 = yphi + yphi;

				tau3 = alpha + beta;
				vn_prime = alpha - beta;
				// corner (-0.5,-0.5) : p1
				dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
				// corner (0.5,-0.5) : p2
				dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
				// corner (-0.5,0.5) : p3
				xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
				// corner (0.5,0.5) : p4
				yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


				f_tr0 = min(dxsp,dysp);
				vn_prime = min(xn_prime,yn_prime);
				tau0=max(f_tr0,vn_prime);
				tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
				tau1=min(tau0,tau3);
				tau2=max(tau0,tau3);
				tau0=min(f_tr0,vn_prime);
				tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
				xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
				if(xn_prime<0){
					xn_prime=0;
				}
				v_max=floor(xn_prime);// u_min
				yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
				zne_max=ceil(yn_prime);// u_max
				if(zne_max>uDetectorPixelNb_GPU-1){
					zne_max=uDetectorPixelNb_GPU-1;
				}

				// the cell lies in the footprint
				f_tr0=0.0f;
				// transaxial footprint
				if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
					dxsp = un_prime-0.5f;
					dysp = un_prime+0.5f;

					xn_prime=max(dxsp,tau0)-tau0;//s1
					yn_prime=min(dysp,tau1)-tau0;//s2
					if(xn_prime<yn_prime){
						f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
					}
					xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
					if(xn_prime> 0){
						f_tr0+=xn_prime;
					}
					xn_prime=max(dxsp,tau2)-tau3;//s1
					yn_prime=min(dysp,tau3)-tau3;//s2
					if(xn_prime<yn_prime){
						f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
					}
					//A2 method
					f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

					for (vn_e = 0;vn_e < K; vn_e++){
						proj[id]+=f_tr0*f_ax0[id];
						f_ax0[id] = 0;
						id += BLOCK_SIZE;
					}
					id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

				}

				if (f_tr0 == 0){
					for (vn_e = 0;vn_e < K; vn_e++){
						f_ax0[id] = 0;
						id += BLOCK_SIZE;
					}
					id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
				}

			}

		}
		//update projections

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		*
		* third and last duplet
		*
		* *****************/
		if (abs(pas_x)< 1.0f || abs(pas_y) <1.0f){
			xphi=roundf(x_e)+ceilf(1.0f-abs(pas_x));
			yphi=roundf(y_e)+ceilf(1.0f-abs(pas_y));


			if (xphi < 0.){
				xn_e = 0;
			}
			else if (xphi > (unsigned int) xVolumePixelNb_GPU-1){
				xn_e = (unsigned int) xVolumePixelNb_GPU-1;
			}
			else{
				xn_e = (unsigned int) xphi;
			}

			if (yphi < 0.){
				yn_e = 0;
			}
			else if (yphi > (unsigned int) yVolumePixelNb_GPU-1){
				yn_e = (unsigned int) yVolumePixelNb_GPU-1;
			}
			else{
				yn_e = (unsigned int) yphi;
			}

			// compute footprint
			xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
			yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

			#if FOV_CYLINDER
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0) && ((4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU))))
			#else
			if ((xphi+1.0f<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1.0f<=float(yVolumePixelNb_GPU) && yphi>=0))
			#endif
			{
				// in the field-of-view
				xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
				yphi=-beta*xn_prime+alpha*yn_prime;

				// compute axial footprint
				// axial footprint : calcul de z pour lesquels F2 non nul
				tau1 = vn_prime-0.5f;
				tau2 = xphi*temp1;
				xn_prime=zVolumeCenterPixel_GPU-0.5f+tau1*tau2;//-0.5 : z_moins
				yn_prime=zVolumeCenterPixel_GPU+0.5f+(tau1+K)*tau2;//+0.5 : z_plus

				if(xn_prime<0.){
					xn_prime=0.;
				}

				v_max=floor(xn_prime);//z_min
				zne_max=ceil(yn_prime);

				if(zne_max> (unsigned int) zVolumePixelNb_GPU-1){
					zne_max= (unsigned int) zVolumePixelNb_GPU-1;
				}

				//axial footprint

				if ((xn_prime+1.0<=float(zVolumePixelNb_GPU) && yn_prime>=0)){
					f_tr0 = float(v_min) - (vDetectorCenterPixel_GPU+0.5f);
					tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
					xn_prime = float(v_max) - (zVolumeCenterPixel_GPU+0.5);
					tau0 = tau3*xn_prime;//chi0
					tau1 = tau3*(xn_prime+1);//chi1
					tau2 = vDetectorCenterPixel_GPU-0.5f+tau0;//v_moins
					yn_prime = vDetectorCenterPixel_GPU+0.5f+tau1;//v_plus
					for(zn_e=v_max;zn_e<=zne_max;zn_e++){

						//bornes
						vn_e = floor(tau2);
						if (vn_e < v_min){
							vn_e = v_min;
						}
						vn_e -= v_min;
						v_max = ceil(yn_prime)-v_min;

						if (v_max > K){
							v_max = K;
						}
						vn_prime = float(vn_e) + f_tr0;
						xn_prime =tex3D(volume_tex,xn_e+0.5f,yn_e+0.5f,zn_e+zn_start+0.5f);
						dxsp = vn_prime+1.0f;
						for (;vn_e < v_max; vn_e++){
							dysp=min(dxsp,tau1)-max(vn_prime,tau0);//s2
							if(dysp > 0){
								f_ax0[id + BLOCK_SIZE*vn_e]+=dysp*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
							}
							vn_prime += 1.0f;
							dxsp += 1.0f;
						}
						yn_prime += tau3;
						tau2 += tau3;
						tau0 += tau3;
						tau1 += tau3;
					}


					//compute transaxial footprint

					// projections
					// project the four corners and build the trapezoid
					/*
					// corner (-0.5,-0.5) : p1
					dxsp=tau0*(yphi+0.5*(beta-alpha))/(xphi+0.5*(-alpha-beta));
					// corner (0.5,-0.5) : p2
					dysp=tau0*(yphi+0.5*(-beta-alpha))/(xphi+0.5*(alpha-beta));
					// corner (-0.5,0.5) : p3
					xn_prime=tau0*(yphi+0.5*(beta+alpha))/(xphi+0.5*(-alpha+beta));
					// corner (0.5,0.5) : p4
					yn_prime=tau0*(yphi+0.5*(-beta+alpha))/(xphi+0.5*(alpha+beta));
					*/

					tau1 = xphi + xphi;
					tau2 = yphi + yphi;

					tau3 = alpha + beta;
					vn_prime = alpha - beta;
					// corner (-0.5,-0.5) : p1
					dxsp=temp2*(tau2-vn_prime)/(tau1-tau3);
					// corner (0.5,-0.5) : p2
					dysp=temp2*(tau2-tau3)/(tau1+vn_prime);
					// corner (-0.5,0.5) : p3
					xn_prime=temp2*(tau2+tau3)/(tau1-vn_prime);
					// corner (0.5,0.5) : p4
					yn_prime=temp2*(tau2+vn_prime)/(tau1+tau3);


					f_tr0 = min(dxsp,dysp);
					vn_prime = min(xn_prime,yn_prime);
					tau0=max(f_tr0,vn_prime);
					tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
					tau1=min(tau0,tau3);
					tau2=max(tau0,tau3);
					tau0=min(f_tr0,vn_prime);
					tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));
					xn_prime=uDetectorCenterPixel_GPU-0.5f+tau0;//-0.5 : u_moins
					if(xn_prime<0){
						xn_prime=0;
					}
					v_max=floor(xn_prime);// u_min
					yn_prime=uDetectorCenterPixel_GPU+0.5f+tau3;//+0.5 : u_plus
					zne_max=ceil(yn_prime);// u_max
					if(zne_max>uDetectorPixelNb_GPU-1){
						zne_max=uDetectorPixelNb_GPU-1;
					}

					// the cell lies in the footprint
					f_tr0=0.0f;
					// transaxial footprint
					if((un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0)){
						dxsp = un_prime-0.5f;
						dysp = un_prime+0.5f;

						xn_prime=max(dxsp,tau0)-tau0;//s1
						yn_prime=min(dysp,tau1)-tau0;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
						}
						xn_prime=min(dysp,tau2)-max(dxsp,tau1);//s1
						if(xn_prime> 0){
							f_tr0+=xn_prime;
						}
						xn_prime=max(dxsp,tau2)-tau3;//s1
						yn_prime=min(dysp,tau3)-tau3;//s2
						if(xn_prime<yn_prime){
							f_tr0+=0.5f*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2);
						}
						//A2 method
						f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xphi*alpha-yphi*beta),abs(yn_prime=xphi*beta+yphi*alpha))));

						for (vn_e = 0;vn_e < K; vn_e++){
							proj[id]+=f_tr0*f_ax0[id];
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;

					}

					if (f_tr0 == 0){
						for (vn_e = 0;vn_e < K; vn_e++){
							f_ax0[id] = 0;
							id += BLOCK_SIZE;
						}
						id = threadIdx.x+(threadIdx.y+threadIdx.z*blockDim.y)*blockDim.x;
					}

				}

			}
			//update projections
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************************
		* steps in x- and y-directions
		* *****************************/
		x_e+=pas_x;
		y_e+=pas_y;
	}

	// méthode A2
	un_prime=un_prime*uDetectorPixelSize_GPU;
	f_tr0 = 1/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU);
	vn_prime=(float(v_min)-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	//un_e = un_e+(v_min)*uDetectorPixelNb_GPU+zne_max*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;
	//un_e = un_e+((v_min-vn_start)*K)*uDetectorPixelNb_GPU+zne_max*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;
	un_e = (threadIdx.x+blockIdx.x*blockDim.x)+(v_min-vn_start)*(blockDim.x*gridDim.x)+zne_max*(blockDim.x*gridDim.x)*(blockDim.y*gridDim.y*K);
	//un_e=(unsigned long long int)(threadIdx.x+blockIdx.x*blockDim.x+(threadIdx.y+blockIdx.y*blockDim.y)*blockDim.x*gridDim.x)+(unsigned long long int)(zne_max)*(unsigned long long int)(blockDim.x*gridDim.x)*(unsigned long long int)(blockDim.y*gridDim.y*K);
	for (v_max = 0; v_max < K; v_max++){
		sinogram_g[un_e]=__fsqrt_rn(1+vn_prime*vn_prime*f_tr0)*proj[id + BLOCK_SIZE*v_max];
		vn_prime += vDetectorPixelSize_GPU;
		un_e += (blockDim.x*gridDim.x);
	}
	//}
}

#endif
