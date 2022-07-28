 /*
 * backprojection3D_kernel.cl
 *
 *      Author: diakite
 */
 
 //#ifndef _BACKPROJECTION3D_KERNEL_CL_
 //#define _BACKPROJECTION3D_KERNEL_CL_

#define PHI_MAX 1024
#define XN_MAX 1024
#define YN_MAX 1024
#define ZN_MAX 256
#define UN_MAX 1024
#define VN_MAX 256

#define XBLOCK_SIZE 64
#define YBLOCK_SIZE 64 
#define ZBLOCK_SIZE 4
 
#define taille 256
#define tailley 8

#define g_wn 98.0f
#define g_vn 98.0f
#define un_0 511.5f
#define vn_0 127.5f


//#pragma once

#undef  TEST_FOV_CYLINDER
#define MARGE_FOV 0
 
typedef struct struct_sampling_opencl {
  //! Taille du voxel du volume en x
  float delta_xn;  

  float xn_0;
  //! (N_yn_FOV / 2) - 0.5
  float yn_0;
  //! (N_zn_FOV / 2) - 0.5
  float zn_0;   ///15

  int N_un;
  //!  Nombre de pixels du plan detecteur en v
  int N_vn;

  int N_xn_FOV;
} type_struct_sampling_opencl;

//#define PHI_MAX 1000
typedef struct struct_constante_opencl {
  float alpha_wn[PHI_MAX];
  float beta_wn[PHI_MAX];
  float gamma_wn;
  float gamma_vn;
  float D;
  float delta_un;
  float gamma_D;//20
  float UN_0;
  float VN_0;
}type_struct_constante_opencl;
 

uint modulo(uint x, uint y){
    return ((long)x * (long)(y)) >> 32; // Easy way to approximate the modulo function
}



//Back proj NDrange CPU and GPU
__kernel void backprojection3D_NDR(
        __global float * restrict volume,
        __global float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
       // __constant type_struct_constante_opencl * restrict constante,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
   // float2 c_ab_wn[PHI_MAX]; // Pour contenir constante alpha et beta

    float s_xn_0 = sampling->xn_0;
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

	//float g_wn = constante->gamma_wn;
	//float un_0 = constante->un_0;
	//float vn_0 = constante->vn_0;
	//float g_vn = constante->gamma_vn;
    //printf("%f\n", sampling->N_un);

    //#pragma unroll PHI_MAX 
    
    //printf("Kernel execution\n");
    xn = get_global_id(0);
    yn = get_global_id(1);
    zn = get_global_id(2);
   
    //printf("%d\n", xn); 
    xn_prime = (float) (xn - s_xn_0);
    yn_prime = (float) (yn - s_yn_0);
    zn_prime = (float) (zn - s_zn_0);
    //printf("%f\n", s_d_xn);

    // float voxel_sum;
    voxel_sum = 0.0f;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //float voxel_sum_tab[PHI_MAX]; // partition en 4 

    //#pragma unroll 32
    for (phi = 0; phi < PHI_MAX; phi++){
        //printf("%f %f\n", alpha_beta[xn].s1, voxel_sum);
        float sinogram_coeff = 0;
        //wn_inverse = 1.0 / ( (constante->alpha_wn[phi] * xn_prime + constante->beta_wn[phi] * yn_prime) * s_d_xn + g_wn);
        //un_prime = (yn_prime * constante->alpha_wn[phi] - xn_prime * constante->beta_wn[phi]) * cntTmp * wn_inverse;
        wn_inverse = 1.0 / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
        un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
        un = un_prime + un_0;
        vn_prime = g_vn * zn_prime * wn_inverse;
        vn = vn_prime + vn_0;
        
        if ( (un >= 0) && (un < s_d_un - 1) && (vn >= 0) && (vn < s_d_vn - 1) ) {
            un_e = (int) trunc(un);
            vn_e = (int) trunc(vn);
            sinogram_coeff = 1; 
            
            //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
        }
        else //un ou vn en dehors des bornes
        {
            
            if ((un >= 0)&&(un < s_d_un - 1 )) 
            {
                
                un_e = (int) trunc(un);

                if ( ((vn >= -0.5) && (vn <= 0)) || ((vn <= s_d_vn - 0.5) && (vn > 0)) ) {
                    vn_e = 0;
                    sinogram_coeff = 1;
                }
            }
            else {
                
                if ((vn >= 0)&&(vn < s_d_vn - 1 )) { 
                    vn_e = (int) trunc(vn);
                    
                    if ( ((un >= -0.5) && (un <= 0)) || ((un <= s_d_un - 0.5) && (un > 0)) ) {
                        un_e = 0;
                        sinogram_coeff = 1;
                        
                    }
                }
            }
        }
            
        voxel_sum += sinogram[(un_e%UN_MAX + (vn_e%VN_MAX) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
        //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
        //if(sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)%256]!=0.0)
            //printf("%f\t", sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)%256]);
        //printf("%f %f\n", alpha_beta[xn].s1, voxel_sum);
        //printf("%f %f\t", voxel_sum, sinogram[(un_e & 255) + (vn_e & 255) * UN_MAX + phi * UN_MAX * VN_MAX]);
        //voxel_sum_tab[phi] = sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX] * sinogram_coeff;
        
    
    }

    volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum;
    
           
}
 


__kernel void backprojection3D_SWI(
         __global float * restrict volume,
        __global float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
       // __constant type_struct_constante_opencl * restrict constante,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
    float2 c_ab_wn[PHI_MAX]; // Pour contenir constante alpha et beta

    float s_xn_0 = sampling->xn_0;
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

	// float g_wn = constante->gamma_wn;
	// float un_0 = constante->un_0;
	// float vn_0 = constante->vn_0;
	// float g_vn = constante->gamma_vn;
    // printf("%f\n", sampling->N_vn);

    //#pragma unroll PHI_MAX 
    // for (unsigned int i = 0; i < PHI_MAX; i++) {
    // 	c_ab_wn[i] = (float2) (constante->alpha_wn[i], constante->beta_wn[i]);
    //    // printf("%f %f\t", constante->alpha_wn[i], constante->beta_wn[i]);
    // }
    //printf("Kernel execution\n");
    for (zn = 0; zn < ZN_MAX; zn++) {
        for (yn = 0; yn < YN_MAX; yn++){
        	//#pragma max_concurrency XN_MAX
            for (xn = 0; xn < XN_MAX; xn++) {
                xn_prime = (float) (xn - s_xn_0);
                yn_prime = (float) (yn - s_yn_0);
        	    zn_prime = (float) (zn - s_zn_0);
                //printf("%f\n", xn_prime);

                float voxel_sum;
                voxel_sum = 0;

                //float voxel_sum_tab[PHI_MAX]; // partition en 4 

                //#pragma unroll 32
                for (phi = 0; phi < PHI_MAX; phi++){

                    float sinogram_coeff = 0;
                    // wn_inverse = 1.0 / ( (c_ab_wn[phi].s0 * xn_prime + c_ab_wn[phi].s1 * yn_prime) * s_d_xn + g_wn);
                    // un_prime = (yn_prime * c_ab_wn[phi].s0 - xn_prime * c_ab_wn[phi].s1) * cntTmp * wn_inverse;
                    wn_inverse = 1.0 / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
                    un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
		            un = un_prime + un_0;
                    vn_prime = g_vn * zn_prime * wn_inverse;
                    vn = vn_prime + vn_0;

                    if ( (un >= 0) && (un < s_d_un - 1) && (vn >= 0) && (vn < s_d_vn - 1) ) {
                        un_e = (int) trunc(un);
                        vn_e = (int) trunc(vn);
                        sinogram_coeff = 1;
                        //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
                    }
                    else //un ou vn en dehors des bornes
                    {
                        if ((un >= 0)&&(un < s_d_un - 1 ))
                        {
                            un_e = (int) trunc(un);

                            if ( ((vn >= -0.5) && (vn <= 0)) || ((vn <= s_d_vn - 0.5) && (vn > 0)) ) {
                                vn_e = 0;
                                sinogram_coeff = 1;
                            }
                        }
                        else {
                            if ((vn >= 0)&&(vn < s_d_vn - 1 )) { 
                                vn_e = (int) trunc(vn);
                                
                                if ( ((un >= -0.5) && (un <= 0)) || ((un <= s_d_un - 0.5) && (un > 0)) ) {
                                    un_e = 0;
                                    sinogram_coeff = 1;
                                }
                            }
                        }
                    }

                    voxel_sum += sinogram[(un_e & 1023) + (vn_e & 255) * UN_MAX + phi * UN_MAX * VN_MAX] * sinogram_coeff;
                    //printf("%f %f\t", voxel_sum, sinogram[(un_e & 255) + (vn_e & 255) * UN_MAX + phi * UN_MAX * VN_MAX]);
                    //voxel_sum_tab[phi] = sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX] * sinogram_coeff;
                    
                
                }
                // 128 -> 102 % d'utilisation
                //#pragma unroll 64
                //for (unsigned int i = 0; i < 256; i++) {
                //	voxel_sum += voxel_sum_tab[i];
                //}

                volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
            }
        }
    }        
}