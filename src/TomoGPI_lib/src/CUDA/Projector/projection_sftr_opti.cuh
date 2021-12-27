/* SFTR projector
*
* Author : Camille Chapdelaine
*/

#ifndef _PROJECTION_SFR_OPTI_H_
#define _PROJECTION_SFR_OPTI_H_

#define K 4

//Projector ! A tester le bon
// projector using shared memory
template <typename T>
__global__ void projection_sftr_generalized_kernel_opti(T* sinogram_g)
{
	// volume copié sur la mémoire de texture
	unsigned int un_e, vn_e, v_min,v_max,zne_max;
	float xn_prime,yn_prime,un_prime,vn_prime;
	float xphi,yphi, dxsp,dysp;
	unsigned int xn_e,yn_e,zn_e,id;
	float tau0, tau1, tau2, tau3, f_tr0;
	bool in_u;
	float pas_x, pas_y;
	unsigned int ksi;
	unsigned long int ksi_max;
	float alpha, beta;


	un_e = threadIdx.x+blockIdx.x*blockDim.x;
	vn_e = (threadIdx.y+blockIdx.y*blockDim.y)*K;//+vn_start;
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	//adress=un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;
	//printf("%llu",adress);
	un_prime=float(un_e)-uDetectorCenterPixel_GPU;
	vn_prime=float(vn_e)-vDetectorCenterPixel_GPU;

	v_min = vn_e;

	id = threadIdx.x+threadIdx.y*blockDim.x;


	// transaxial footprint in shared memory
	__shared__ float f_tr[32];
	__shared__ float f_ax0[256*K];
	__shared__ float proj[256*K];
	__shared__ float xy[2*256];

	//initialization
	for(vn_e = 0; vn_e<K; vn_e++){
		proj[id+vn_e*256] = 0;
		f_ax0[id+vn_e*256] = 0;
	}

	alpha = alphaIOcylinderC_GPU[zne_max];
	beta = betaIOcylinderC_GPU[zne_max];

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alpha-un_prime*uDetectorPixelSize_GPU*beta;

	//y_p-y_s
	dysp=focusDetectorDistance_GPU*beta+un_prime*uDetectorPixelSize_GPU*alpha;

	// initialize x and y indices
	tau0 = (focusObjectDistance_GPU/xVolumePixelSize_GPU);
	if(abs(dxsp)>=abs(dysp))
	{
		pas_x=1.0;
		pas_y=dysp/dxsp;
		ksi_max=xVolumePixelNb_GPU;
		xy[id]=0.0;
		xy[id+256]=yVolumeCenterPixel_GPU-tau0*beta-pas_y*(xVolumeCenterPixel_GPU-tau0*alpha);
	}
	else
	{
		pas_x=dxsp/dysp;
		pas_y=1.0;
		ksi_max=yVolumePixelNb_GPU;
		xy[id+256]=0.0;
		xy[id]=xVolumeCenterPixel_GPU-tau0*alpha-pas_x*(yVolumeCenterPixel_GPU-tau0*beta);
	}

	// initialize projection
	// loop on ksi
	for(ksi=0;ksi<ksi_max;ksi++){

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * first duplet
		 *
		 * *****************/

		xphi=roundf(xy[id])-ceilf(1-abs(pas_x));
		yphi=roundf(xy[id+256])-ceilf(1-abs(pas_y));


		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}

		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u && (abs(pas_x)< 1 || abs(pas_y) <1));

		//axial footprint
		if (in_u){
			tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
			xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
			tau0 = tau3*xn_prime;//chi0
			tau1 = tau3*(xn_prime+1);//chi1
			tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
			yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
			for(zn_e=v_max;zn_e<=zne_max;zn_e++){
				xn_prime =tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
				//bornes
				vn_e = floor(tau2);
				if (vn_e <v_min){
					vn_e = v_min;
				}
				vn_e -= v_min;
				v_max = ceil(yn_prime)-v_min;

				if (v_max > K){
					v_max = K;
				}
				vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;

				for (;vn_e < v_max; vn_e++){
					dxsp=max((vn_prime-0.5),tau0);//s1
					dysp=min((vn_prime+0.5),tau1);//s2
					if(dxsp<dysp){
						f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
					}
					vn_prime += 1;
				}
				yn_prime += tau3;
				tau2 += tau3;
				tau0 += tau3;
				tau1 += tau3;
			}
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			vn_prime = min(xn_prime,yn_prime);
			tau0=max(min(dxsp,dysp),vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			if (in_u){
				vn_prime=0.0;
				dxsp = un_prime - 0.5;
				dysp = un_prime + 0.5;
				xn_prime=max(dxsp,tau0)-tau0;//s1
				yn_prime=min(dysp,tau1)-tau0;//s2
				if(xn_prime<yn_prime){
					vn_prime+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
				}
				xn_prime=max(dxsp,tau1);//s1
				yn_prime=min(dysp,tau2);//s2
				if(xn_prime<yn_prime){
					vn_prime+=(yn_prime-xn_prime);
				}
				xn_prime=max(dxsp,tau2)-tau3;//s1
				yn_prime=min(dysp,tau3)-tau3;//s2
				if(xn_prime<yn_prime){
					vn_prime+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
				}
				//A2 method
				xn_prime=xphi*alpha-yphi*beta;//cos_psi
				yn_prime=xphi*beta+yphi*alpha;//sin_psi
				f_tr[threadIdx.x]=vn_prime*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			}
		}
		__syncthreads();

		//update projections
		xn_prime = f_tr[threadIdx.x];
		for (vn_e = 0;vn_e < K; vn_e++){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			f_ax0[id + 256*(vn_e)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * second duplet
		 *
		 * *****************/
		xphi=roundf(xy[id]);
		yphi=roundf(xy[id+256]);

		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}
		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}



		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}

		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u && (abs(pas_x)< 1 || abs(pas_y) <1));

		if (in_u){
			tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
			xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
			tau0 = tau3*xn_prime;//chi0
			tau1 = tau3*(xn_prime+1);//chi1
			tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
			yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
			for(zn_e=v_max;zn_e<=zne_max;zn_e++){
				xn_prime =tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
				//bornes
				vn_e = floor(tau2);
				if (vn_e <v_min){
					vn_e = v_min;
				}
				vn_e -= v_min;
				v_max = ceil(yn_prime)-v_min;

				if (v_max > K){
					v_max = K;
				}
				vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;

				for (;vn_e < v_max; vn_e++){
					dxsp=max((vn_prime-0.5),tau0);//s1
					dysp=min((vn_prime+0.5),tau1);//s2
					if(dxsp<dysp && in_u){
						f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
					}
					vn_prime += 1;
				}
				yn_prime += tau3;
				tau2 += tau3;
				tau0 += tau3;
				tau1 += tau3;
			}
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			vn_prime = min(xn_prime,yn_prime);
			tau0=max(min(dxsp,dysp),vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			if (in_u){
				vn_prime=0.0;
				dxsp = un_prime - 0.5;
				dysp = un_prime + 0.5;
				xn_prime=max(dxsp,tau0)-tau0;//s1
				yn_prime=min(dysp,tau1)-tau0;//s2
				if(xn_prime<yn_prime){
					vn_prime+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
				}
				xn_prime=max(dxsp,tau1);//s1
				yn_prime=min(dysp,tau2);//s2
				if(xn_prime<yn_prime){
					vn_prime+=(yn_prime-xn_prime);
				}
				xn_prime=max(dxsp,tau2)-tau3;//s1
				yn_prime=min(dysp,tau3)-tau3;//s2
				if(xn_prime<yn_prime){
					vn_prime+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
				}
				//A2 method
				xn_prime=xphi*alpha-yphi*beta;//cos_psi
				yn_prime=xphi*beta+yphi*alpha;//sin_psi
				f_tr[threadIdx.x]=vn_prime*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			}
		}
		__syncthreads();


		//update projections
		xn_prime = f_tr[threadIdx.x];
		for (vn_e = 0;vn_e < K; vn_e++){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			f_ax0[id + 256*(vn_e)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * third and last duplet
		 *
		 * *****************/
		xphi=roundf(xy[id])+ceilf(1-abs(pas_x));
		yphi=roundf(xy[id+256])+ceilf(1-abs(pas_y));



		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}


		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max>zVolumePixelNb_GPU-1){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u && (abs(pas_x)< 1 || abs(pas_y) <1));

		if (in_u){
			tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
			xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
			tau0 = tau3*xn_prime;//chi0
			tau1 = tau3*(xn_prime+1);//chi1
			tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
			yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
			for(zn_e=v_max;zn_e<=zne_max;zn_e++){
				xn_prime =tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
				//bornes
				vn_e = floor(tau2);
				if (vn_e <v_min){
					vn_e = v_min;
				}
				vn_e -= v_min;
				v_max = ceil(yn_prime)-v_min;

				if (v_max > K){
					v_max = K;
				}
				vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;

				for (;vn_e < v_max; vn_e++){
					dxsp=max((vn_prime-0.5),tau0);//s1
					dysp=min((vn_prime+0.5),tau1);//s2
					if(dxsp<dysp && in_u){
						f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
					}
					vn_prime += 1;
				}
				yn_prime += tau3;
				tau2 += tau3;
				tau0 += tau3;
				tau1 += tau3;
			}
		}

		//compute transaxial footprint
		if(threadIdx.y==0){
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			vn_prime = min(xn_prime,yn_prime);
			tau0=max(min(dxsp,dysp),vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(min(dxsp,dysp),vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max>uDetectorPixelNb_GPU-1){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			if (in_u){
				vn_prime=0.0;
				dxsp = un_prime - 0.5;
				dysp = un_prime + 0.5;
				xn_prime=max(dxsp,tau0)-tau0;//s1
				yn_prime=min(dysp,tau1)-tau0;//s2
				if(xn_prime<yn_prime){
					vn_prime+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
				}
				xn_prime=max(dxsp,tau1);//s1
				yn_prime=min(dysp,tau2);//s2
				if(xn_prime<yn_prime){
					vn_prime+=(yn_prime-xn_prime);
				}
				xn_prime=max(dxsp,tau2)-tau3;//s1
				yn_prime=min(dysp,tau3)-tau3;//s2
				if(xn_prime<yn_prime){
					vn_prime+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
				}
				//A2 method
				xn_prime=xphi*alpha-yphi*beta;//cos_psi
				yn_prime=xphi*beta+yphi*alpha;//sin_psi
				f_tr[threadIdx.x]=vn_prime*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));
			}
		}
		__syncthreads();


		//update projections
		xn_prime = f_tr[threadIdx.x];
		for (vn_e = 0;vn_e < K; vn_e++){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			f_ax0[id + 256*(vn_e)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************************
		 * steps in x- and y-directions
		 * *****************************/
		xy[id]+=pas_x;
		xy[id+256]+=pas_y;
	}

	// méthode A2
	un_prime=un_prime*uDetectorPixelSize_GPU;
	tau1 = 1/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU);
	vn_prime=(float(v_min)-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;
	zne_max = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	un_e = un_e+v_min*uDetectorPixelNb_GPU+zne_max*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;;
	for (v_max = 0; v_max < K; v_max++){
		sinogram_g[un_e+v_max*uDetectorPixelNb_GPU]=proj[id + 256*v_max]*__fsqrt_rn(1+vn_prime*vn_prime*tau1);
		vn_prime += vDetectorPixelSize_GPU;
	}
}

//test
template <typename T>
__global__ void projection_sftr_generalized_kernel_opti_v2(T* sinogram_g)
{
	// volume copié sur la mémoire de texture
	unsigned long int phi,un_e, vn_e, v_min,v_max,zne_max;
	float xn_prime,yn_prime,un_prime,vn_prime;
	float xphi,yphi, dxsp,dysp;
	unsigned int xn_e,yn_e,zn_e,id;
	float x_e,y_e;
	float tau0, tau1, tau2, tau3, f_tr0;
	bool in_u;
	float pas_x, pas_y;
	unsigned int ksi;
	unsigned long int ksi_max;
	float alpha, beta;
	// transaxial footprint in shared memory
	float f_tr;


	un_e = threadIdx.x+blockIdx.x*blockDim.x;
	vn_e = (threadIdx.y+blockIdx.y*blockDim.y)*K;//+vn_start;
	phi = threadIdx.z + blockIdx.z*blockDim.z;//+phi_start;
	//adress=un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU;
	//printf("%llu",adress);
	un_prime=float(un_e)-uDetectorCenterPixel_GPU;
	vn_prime=float(vn_e)-vDetectorCenterPixel_GPU;

	v_min = vn_e;

	id = threadIdx.x+threadIdx.y*blockDim.x;

	__shared__ float f_ax0[256*K];
	__shared__ float proj[256*K];

	//initialization
	for(vn_e = 0; vn_e<K; vn_e++){
		proj[id+vn_e*256] = 0;
		f_ax0[id+vn_e*256] = 0;
	}

	alpha = alphaIOcylinderC_GPU[phi];
	beta = betaIOcylinderC_GPU[phi];

	//x_p-x_s
	dxsp=focusDetectorDistance_GPU*alpha-un_prime*uDetectorPixelSize_GPU*beta;
	//y_p-y_s
	dysp=focusDetectorDistance_GPU*beta+un_prime*uDetectorPixelSize_GPU*alpha;

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

	// loop on ksi
	for(ksi=0;ksi<ksi_max;ksi++){

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * first duplet
		 *
		 * *****************/

		xphi=roundf(x_e)-ceilf(1-abs(pas_x));
		yphi=roundf(y_e)-ceilf(1-abs(pas_y));


		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}

		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max+1>zVolumePixelNb_GPU){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u);

		//axial footprint
		tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
		xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
		tau0 = tau3*xn_prime;//chi0
		tau1 = tau3*(xn_prime+1);//chi1
		tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
		yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
		for(zn_e=v_max;zn_e<=zne_max;zn_e++){
			xn_prime =tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
    		//bornes
			vn_e = floor(tau2);
			if (vn_e <v_min){
				vn_e = v_min;
			}
			vn_e -= v_min;
			v_max = ceil(yn_prime)-v_min;

			if (v_max > K){
				v_max = K;
			}
			vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;

			for (;vn_e < v_max; vn_e++){
				dxsp=max((vn_prime-0.5),tau0);//s1
				dysp=min((vn_prime+0.5),tau1);//s2
				if(dxsp<dysp && in_u && (abs(pas_x)< 1 || abs(pas_y) <1)){
					f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
				}
				vn_prime += 1;
			}
			yn_prime += tau3;
			tau2 += tau3;
			tau0 += tau3;
			tau1 += tau3;
		}

		//compute transaxial footprint
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			f_tr0 = min(dxsp,dysp);
			vn_prime = min(xn_prime,yn_prime);
			tau0=max(f_tr0,vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(f_tr0,vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max+1>uDetectorPixelNb_GPU){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			f_tr0=0.0;
			dxsp = un_prime - 0.5;
			dysp = un_prime + 0.5;
			xn_prime=max(dxsp,tau0)-tau0;//s1
			yn_prime=min(dysp,tau1)-tau0;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
			}
			xn_prime=max(dxsp,tau1);//s1
			yn_prime=min(dysp,tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max(dxsp,tau2)-tau3;//s1
			yn_prime=min(dysp,tau3)-tau3;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alpha-yphi*beta;//cos_psi
			yn_prime=xphi*beta+yphi*alpha;//sin_psi
			f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));




		//update projections
		xn_prime = f_tr0;
		for (vn_e = 0;vn_e < K; vn_e+=2){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			proj[id + 256*(vn_e+1)]+=xn_prime*f_ax0[id + 256*(vn_e+1)];
			f_ax0[id + 256*(vn_e)] = 0;
			f_ax0[id + 256*(vn_e+1)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * second duplet
		 *
		 * *****************/
		xphi=roundf(x_e);
		yphi=roundf(y_e);

		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}
		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}



		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max+1>zVolumePixelNb_GPU){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u && (abs(pas_x)< 1 || abs(pas_y) <1));

		//axial footprint
		tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
		xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
		tau0 = tau3*xn_prime;//chi0
		tau1 = tau3*(xn_prime+1);//chi1
		tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
		yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
		for(zn_e=v_max;zn_e<=zne_max;zn_e++){
			xn_prime = tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
    		//bornes
			vn_e = floor(tau2);
			if (vn_e <v_min){
				vn_e = v_min;
			}
			vn_e -= v_min;
			v_max = ceil(yn_prime)-v_min;

			if (v_max > K){
				v_max = K;
			}
			vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;
			for (;vn_e < v_max; vn_e++){
				dxsp=max((vn_prime-0.5),tau0);//s1
				dysp=min((vn_prime+0.5),tau1);//s2
				if(dxsp<dysp){
					f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
				}
				vn_prime += 1;
			}
			yn_prime += tau3;
			tau2 += tau3;
			tau0 += tau3;
			tau1 += tau3;
		}

		//compute transaxial footprint
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			f_tr0 = min(dxsp,dysp);
			vn_prime = min(xn_prime,yn_prime);
			tau0=max(f_tr0,vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(f_tr0,vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max+1>uDetectorPixelNb_GPU){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			f_tr0=0.0;
			dxsp = un_prime - 0.5;
			dysp = un_prime + 0.5;
			xn_prime=max(dxsp,tau0)-tau0;//s1
			yn_prime=min(dysp,tau1)-tau0;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
			}
			xn_prime=max(dxsp,tau1);//s1
			yn_prime=min(dysp,tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max(dxsp,tau2)-tau3;//s1
			yn_prime=min(dysp,tau3)-tau3;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alpha-yphi*beta;//cos_psi
			yn_prime=xphi*beta+yphi*alpha;//sin_psi
			f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));




		//update projections
		xn_prime = f_tr0;
		for (vn_e = 0;vn_e < K; vn_e+=2){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			proj[id + 256*(vn_e+1)]+=xn_prime*f_ax0[id + 256*(vn_e+1)];
			f_ax0[id + 256*(vn_e)] = 0;
			f_ax0[id + 256*(vn_e+1)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************
		 *
		 * third and last duplet
		 *
		 * *****************/
		xphi=roundf(x_e)+ceilf(1-abs(pas_x));
		yphi=roundf(y_e)+ceilf(1-abs(pas_y));



		if (xphi < 0){
			xn_e = 0;
		}
		else if (xphi > xVolumePixelNb_GPU-1){
			xn_e = xVolumePixelNb_GPU-1;
		}
		else{
			xn_e = (unsigned int) xphi;
		}

		if (yphi < 0){
			yn_e = 0;
		}
		else if (yphi > yVolumePixelNb_GPU-1){
			yn_e = yVolumePixelNb_GPU-1;
		}
		else{
			yn_e = (unsigned int) yphi;
		}


		// compute footprint
		xn_prime=float(xn_e)-xVolumeCenterPixel_GPU;
		yn_prime=float(yn_e)-yVolumeCenterPixel_GPU;

		// in the field-of-view
		in_u=(xphi+1<=float(xVolumePixelNb_GPU) && xphi>=0 && yphi+1<=float(yVolumePixelNb_GPU) && yphi>=0 && (4*(xn_prime*xn_prime+yn_prime*yn_prime)<float(xVolumePixelNb_GPU*xVolumePixelNb_GPU)));

		xphi=(focusObjectDistance_GPU/xVolumePixelSize_GPU)+alpha*xn_prime+beta*yn_prime;
		yphi=-beta*xn_prime+alpha*yn_prime;

		// compute axial footprint
		// axial footprint : calcul de z pour lesquels F2 non nul
		tau2 = xphi*vDetectorPixelSize_GPU/focusDetectorDistance_GPU;
		xn_prime=zVolumeCenterPixel_GPU-0.5+(vn_prime-0.5)*tau2;//-0.5 : z_moins
		yn_prime=zVolumeCenterPixel_GPU+0.5+(vn_prime+float(K)-0.5)*tau2;//+0.5 : z_plus

		if(xn_prime<0){
			xn_prime=0;
		}

		v_max=floor(xn_prime);
		zne_max=ceil(yn_prime);

		if(zne_max+1>zVolumePixelNb_GPU){
			zne_max=zVolumePixelNb_GPU-1;
		}
		in_u=(xn_prime+1<=float(zVolumePixelNb_GPU) && yn_prime>=0  && in_u && (abs(pas_x)< 1 || abs(pas_y) <1));

		//axial footprint
		tau3 = focusDetectorDistance_GPU/(vDetectorPixelSize_GPU*xphi);
		xn_prime = float(v_max) - zVolumeCenterPixel_GPU-0.5;
		tau0 = tau3*xn_prime;//chi0
		tau1 = tau3*(xn_prime+1);//chi1
		tau2 = vDetectorCenterPixel_GPU-0.5+tau0;//v_moins
		yn_prime = vDetectorCenterPixel_GPU+0.5+tau1;//v_plus
		for(zn_e=v_max;zn_e<=zne_max;zn_e++){
			xn_prime = tex3D(volume_tex,xn_e+0.5,yn_e+0.5,zn_e+0.5);
    		//bornes
			vn_e = floor(tau2);
			if (vn_e <v_min){
				vn_e = v_min;
			}
			vn_e -= v_min;
			v_max = ceil(yn_prime)-v_min;

			if (v_max > K){
				v_max = K;
			}


			vn_prime = float(vn_e+v_min) - vDetectorCenterPixel_GPU;
			for (;vn_e < v_max; vn_e++){

				dxsp=max((vn_prime-0.5),tau0);//s1
				dysp=min((vn_prime+0.5),tau1);//s2
				if(dxsp<dysp && in_u){
					f_ax0[id + 256*vn_e]+=(dysp-dxsp)*xn_prime;//d_volume[xn_e+yn_e*xVolumePixelNb_GPU+zn_e*xVolumePixelNb_GPU*yVolumePixelNb_GPU]
				}
				vn_prime += 1;
			}
			yn_prime += tau3;
			tau2 += tau3;
			tau0 += tau3;
			tau1 += tau3;
		}

		//compute transaxial footprint
			// projections
			// project the four corners and build the trapezoid
			tau2 =focusDetectorDistance_GPU/uDetectorPixelSize_GPU;
			tau3 = yphi+yphi;
			tau0 = xphi+xphi;
			// corner (-0.5,-0.5) : p1
			dxsp=tau2*(tau3+beta-alpha)/(tau0-alpha-beta);
			// corner (0.5,-0.5) : p2
			dysp=tau2*(tau3-beta-alpha)/(tau0+alpha-beta);
			// corner (-0.5,0.5) : p3
			xn_prime=tau2*(tau3+beta+alpha)/(tau0-alpha+beta);
			// corner (0.5,0.5) : p4
			yn_prime=tau2*(tau3-beta+alpha)/(tau0+alpha+beta);

			f_tr0 = min(dxsp,dysp);
			vn_prime = min(xn_prime,yn_prime);
			tau0=max(f_tr0,vn_prime);
			tau3=min(max(dxsp,dysp),max(xn_prime,yn_prime));
			tau1=min(tau0,tau3);
			tau2=max(tau0,tau3);
			tau0=min(f_tr0,vn_prime);
			tau3=max(max(dxsp,dysp),max(xn_prime,yn_prime));

			xn_prime=uDetectorCenterPixel_GPU-0.5+tau0;//-0.5 : u_moins
			yn_prime=uDetectorCenterPixel_GPU+0.5+tau3;//+0.5 : u_plus
			if(xn_prime<0){
				xn_prime=0;
			}
			v_max=floor(xn_prime);// u_min
			zne_max=ceil(yn_prime);// u_max

			if(zne_max+1>uDetectorPixelNb_GPU){
				zne_max=uDetectorPixelNb_GPU-1;
			}

			// the cell lies in the footprint
			in_u=(un_e<=zne_max && un_e>=v_max && xn_prime<=float(uDetectorPixelNb_GPU-1) && yn_prime>=0);

			f_tr0=0.0;
			dxsp = un_prime - 0.5;
			dysp = un_prime + 0.5;
			xn_prime=max(dxsp,tau0)-tau0;//s1
			yn_prime=min(dysp,tau1)-tau0;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=0.5*(yn_prime-xn_prime)*(yn_prime+xn_prime)/(tau1-tau0);
			}
			xn_prime=max(dxsp,tau1);//s1
			yn_prime=min(dysp,tau2);//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(yn_prime-xn_prime);
			}
			xn_prime=max(dxsp,tau2)-tau3;//s1
			yn_prime=min(dysp,tau3)-tau3;//s2
			if(xn_prime<yn_prime && in_u){
				f_tr0+=(0.5*(xn_prime-yn_prime)*(xn_prime+yn_prime)/(tau3-tau2));
			}
			//A2 method
			xn_prime=xphi*alpha-yphi*beta;//cos_psi
			yn_prime=xphi*beta+yphi*alpha;//sin_psi
			f_tr0=f_tr0*(__fsqrt_rn(xphi*xphi+yphi*yphi)*xVolumePixelSize_GPU/(max(abs(xn_prime),abs(yn_prime))));




		//update projections
		xn_prime = f_tr0;
		for (vn_e = 0;vn_e < K; vn_e+=2){
			proj[id + 256*vn_e]+=xn_prime*f_ax0[id + 256*vn_e];
			proj[id + 256*(vn_e+1)]+=xn_prime*f_ax0[id + 256*(vn_e+1)];
			f_ax0[id + 256*(vn_e)] = 0;
			f_ax0[id + 256*(vn_e+1)] = 0;
		}

		vn_prime=float(v_min)-vDetectorCenterPixel_GPU;

		/*******************************
		 * steps in x- and y-directions
		 * *****************************/
		x_e=x_e+pas_x;
		y_e=y_e+pas_y;
	}

	// méthode A2
	un_prime=un_prime*uDetectorPixelSize_GPU;
	v_max = v_min + K;
	f_tr0 = 1/(un_prime*un_prime+focusDetectorDistance_GPU*focusDetectorDistance_GPU);
	vn_prime=(float(v_min)-vDetectorCenterPixel_GPU)*vDetectorPixelSize_GPU;
	for (vn_e = v_min; vn_e < v_max; vn_e++){
		sinogram_g[un_e+vn_e*uDetectorPixelNb_GPU+phi*uDetectorPixelNb_GPU*vDetectorPixelNb_GPU]=proj[id + 256*(vn_e-v_min)]*__fsqrt_rn(1+vn_prime*vn_prime*f_tr0);
		vn_prime += vDetectorPixelSize_GPU;
	}
}

#endif
