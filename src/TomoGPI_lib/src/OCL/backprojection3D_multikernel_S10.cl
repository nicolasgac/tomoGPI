#define PHI_MAX 256
#define XN_MAX 256
#define YN_MAX 256
#define ZN_MAX 256
#define UN_MAX 256
#define VN_MAX 256

#define XBLOCK_SIZE 64
#define YBLOCK_SIZE 64 
#define ZBLOCK_SIZE 8
 
#define taille 256
#define tailley 12
 
#define g_wn 98.0f
#define g_vn 98.0f
#define un_0 127.5f
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
 

void fecth(__global float * vol, __local float * mem_local, int u, int v, int phi){

    //__local float mem_local[24*8];

    for(uint i = 0; i < 8; i++){
        for(uint j = 0; j < 24; j++){
            mem_local[i * 8 + j] = vol[u + v * 256 + phi * 256 * 256];
            u++;
            v++;
        }
    }

}


uint modulo(uint x, uint y){
    return ((long)x * (long)y) >> 32;
}

__attribute__((scheduler_target_fmax_mhz(250)))
__kernel void backprojection3D_0(
        __global float * restrict volume,
        __global const volatile float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
    
    float __attribute__((numbanks(1)))
    	  __attribute__((doublepump))
    	  __attribute__((numwriteports(1)))
 	      __attribute__((numreadports(64))) cached_sino[taille*tailley];
 	 
 	  
    float __attribute__((numbanks(1)))
    	  __attribute__((numwriteports(2)))
 	      __attribute__((numreadports(2))) tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
   
 
    float s_xn_0 = sampling->xn_0; 
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

    
    int temp;

	
    #pragma loop_coalesce 3
    for(uint blockz = 0; blockz < 64; blockz+=ZBLOCK_SIZE){
        for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
            for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
                mean_global=0.0;

		#pragma unroll 64
                for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
                    tuile_voxel[ptr]=0.0f;
                }

		#pragma max_concurrency 4
                for (phi = 0; phi < PHI_MAX; phi++){
                    float xn0, yn0, zn0;
                    xn0 = (float) (blockx - s_xn_0);
                    yn0 = (float) (blocky - s_yn_0);
                    zn0 = (float) (blockz - s_zn_0);
                    float wn0 = 1.0f / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
                    uint un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
                    uint vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
                    
                    uint un0_pr=0;
                    uint vn0_pr=0;
                    
                   // Zone memoire à precharger dependant de un et vn
                    float indice = ((vn0 + tailley/2)/(tailley));
                
                    vn0_pr = (abs(vn0 - 2))&255; //ramener vn au multiple proche de la hauteur du tableau
                    
                    uint temp = vn0_pr;
                    
                    for(int m=0; m<tailley; m++){
                        un0_pr=0;
                        #pragma unroll 8
                        for(int n=0; n<taille; n++){
                            cached_sino[m*taille +n] = sinogram[(n + (vn0_pr) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                            //un0_pr =(un0_pr + 1);
                            
                        }
                        vn0_pr =(vn0_pr + 1)&(UN_MAX - 1);
                    }
                    
                    for (int zz = 0; zz < ZBLOCK_SIZE; zz++) {
                    	
                        for (int yy = 0; yy < YBLOCK_SIZE; yy++) {
                            
                            
                            #pragma unroll 
                            for (int xx = 0; xx < XBLOCK_SIZE; xx++) { 
                    		
                                xn = xx + blockx;
                                yn = yy + blocky;
                                zn = zz + blockz;
                    
                                
                                xn_prime = (float) (xn - s_xn_0);
                                yn_prime = (float) (yn - s_yn_0);
                                zn_prime = (float) (zn - s_zn_0);
                                
                                voxel_sum = 0.0f;
                                
		                    float sinogram_coeff = 0;
		                    unsigned short set_un=0, set_vn=0;
		                    
		                    wn_inverse = 1.0f / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
		                    un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
		                    un = un_prime + un_0;
		                    vn_prime = g_vn * zn_prime * wn_inverse;
		                    vn = vn_prime + vn_0;
                                
                                 if ( (un >= 0.0f) && (un < s_d_un - 1.0f) && (vn >= 0.0f) && (vn < s_d_vn - 1.0f) ) {
                                    //un_e = (int) trunc(un);
                                    //vn_e = (int) trunc(vn);
                                    set_un=1;
                                    set_vn=1;
                                    sinogram_coeff = 1; 
                                    
                                    //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
                                }
                                else //un ou vn en dehors des bornes
                                {
                                    
                                    if ((un >= 0.0f)&&(un < s_d_un - 1.0f )) 
                                    {
                                        
                                        //un_e = (int) trunc(un);
                                        set_un=1;

                                        if ( ((vn >= -0.5f) && (vn <= 0.0f)) || ((vn <= s_d_vn - 0.5f) && (vn > 0.0f)) ) {
                                            //vn_e = 0;
                                            set_vn=0;
                                            sinogram_coeff = 1;
                                        }
                                    }
                                    else {
                                        
                                        if ((vn >= 0.0f)&&(vn < s_d_vn - 1.0f )) { 
                                            //vn_e = (int) trunc(vn);
                                            set_vn=1;
                                            
                                            if ( ((un >= -0.5f) && (un <= 0.0f)) || ((un <= s_d_un - 0.5f) && (un > 0.0f)) ) {
                                                //un_e = 0;
                                                set_un=0;
                                                sinogram_coeff = 1;
                                                
                                            }
                                        }
                                    }
                                } 
                                un_e = (unsigned short) (un) * set_un;
                                vn_e = (unsigned short) (vn) * set_vn;
                                
                                
                                uint VN = modulo(vn_e, temp);
                                if(VN ==0 ){
                                    VN= vn_e - temp;
                                }
                                   
                                    //if((vn_e)%(temp) < tailley)
                                    if(VN < tailley)
                                        tuile_voxel[xx + (yy) * XBLOCK_SIZE + (zz) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[((un_e)&(taille)-1) + (VN) * taille]  * sinogram_coeff;
                                    
                                }
  
                                //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
        
                            }
                        } 
                  
                } 
            #pragma loop_coalesce 3
            for(int k=0; k<ZBLOCK_SIZE; k++){
                     for(int j=0; j<YBLOCK_SIZE; j++){
                        #pragma unroll 64
                        for(int i=0; i<XBLOCK_SIZE; i++){
                            volume[(i + blockx) + ((j + blocky)) * XN_MAX + ((k + blockz)) * XN_MAX * YN_MAX] = tuile_voxel[i + (j) * XBLOCK_SIZE + (k) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn; //* s_d_xns
                        }
                    }
                }

                 
            }   
        }       
    }
}


__attribute__((scheduler_target_fmax_mhz(250)))
__kernel void backprojection3D_1(
        __global float * restrict volume,
        __global const volatile float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
   
    float __attribute__((numbanks(1)))
    	  __attribute__((doublepump))
    	  __attribute__((numwriteports(1)))
 	      __attribute__((numreadports(64))) cached_sino[taille*tailley];
 	 
 	  
    float __attribute__((numbanks(1)))
    	 __attribute__((numwriteports(2)))
 	      __attribute__((numreadports(2))) tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
   
 
    float s_xn_0 = sampling->xn_0; 
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

    
    int temp;

    #pragma loop_coalesce 3
    for(uint blockz = 64; blockz < 128; blockz+=ZBLOCK_SIZE){
        for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
            for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
                mean_global=0.0;

		#pragma unroll 64
                for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
                    tuile_voxel[ptr]=0.0f;
                }

		#pragma max_concurrency 4
                for (phi = 0; phi < PHI_MAX; phi++){
                    float xn0, yn0, zn0;
                    xn0 = (float) (blockx - s_xn_0);
                    yn0 = (float) (blocky - s_yn_0);
                    zn0 = (float) (blockz - s_zn_0);
                    float wn0 = 1.0f / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
                    uint un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
                    uint vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
                    
                    uint un0_pr=0;
                    uint vn0_pr=0;
                    
                   // Zone memoire à precharger dependant de un et vn
                    float indice = ((vn0 + tailley/2)/(tailley));
                    
                    vn0_pr = (abs(vn0 - 2))&255; //ramener vn au multiple proche de la hauteur du tableau
                    
                    uint temp = vn0_pr;
                    
                    for(int m=0; m<tailley; m++){
                        un0_pr=0;
                        #pragma unroll 8
                        for(int n=0; n<taille; n++){
                            cached_sino[m*taille +n] = sinogram[(n + (vn0_pr) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                            //un0_pr =(un0_pr + 1);
                            
                        }
                        vn0_pr =(vn0_pr + 1)&(UN_MAX - 1);
                    }
                    
                    for (int zz = 0; zz < ZBLOCK_SIZE; zz++) {
                    	
                        for (int yy = 0; yy < YBLOCK_SIZE; yy++) {
                            
                            
                            #pragma unroll 
                            for (int xx = 0; xx < XBLOCK_SIZE; xx++) { 
                    		
                                xn = xx + blockx;
                                yn = yy + blocky;
                                zn = zz + blockz;
                    
                                
                                xn_prime = (float) (xn - s_xn_0);
                                yn_prime = (float) (yn - s_yn_0);
                                zn_prime = (float) (zn - s_zn_0);
                                
                                voxel_sum = 0.0f;
                                
		                    float sinogram_coeff = 0;
		                    unsigned short set_un=0, set_vn=0;
		                    
		                    wn_inverse = 1.0f / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
		                    un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
		                    un = un_prime + un_0;
		                    vn_prime = g_vn * zn_prime * wn_inverse;
		                    vn = vn_prime + vn_0;
                                
                                 if ( (un >= 0.0f) && (un < s_d_un - 1.0f) && (vn >= 0.0f) && (vn < s_d_vn - 1.0f) ) {
                                    //un_e = (int) trunc(un);
                                    //vn_e = (int) trunc(vn);
                                    set_un=1;
                                    set_vn=1;
                                    sinogram_coeff = 1; 
                                    
                                    //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
                                }
                                else //un ou vn en dehors des bornes
                                {
                                    
                                    if ((un >= 0.0f)&&(un < s_d_un - 1.0f )) 
                                    {
                                        
                                        //un_e = (int) trunc(un);
                                        set_un=1;

                                        if ( ((vn >= -0.5f) && (vn <= 0.0f)) || ((vn <= s_d_vn - 0.5f) && (vn > 0.0f)) ) {
                                            //vn_e = 0;
                                            set_vn=0;
                                            sinogram_coeff = 1;
                                        }
                                    }
                                    else {
                                        
                                        if ((vn >= 0.0f)&&(vn < s_d_vn - 1.0f )) { 
                                            //vn_e = (int) trunc(vn);
                                            set_vn=1;
                                            
                                            if ( ((un >= -0.5f) && (un <= 0.0f)) || ((un <= s_d_un - 0.5f) && (un > 0.0f)) ) {
                                                //un_e = 0;
                                                set_un=0;
                                                sinogram_coeff = 1;
                                                
                                            }
                                        }
                                    }
                                } 
                                un_e = (unsigned short) (un) * set_un;
                                vn_e = (unsigned short) (vn) * set_vn;
                                
                                
                                uint VN = modulo(vn_e, temp);
                                if(VN ==0 ){
                                    VN= vn_e - temp;
                                }
                                   
                                    //if((vn_e)%(temp) < tailley)
                                    if(VN < tailley)
                                        tuile_voxel[xx + (yy) * XBLOCK_SIZE + (zz) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[((un_e)&(taille)-1) + (VN) * taille]  * sinogram_coeff;
                                    
                                }
  
                                //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
        
                            }
                        } 
                  
                } 
            #pragma loop_coalesce 3
            for(int k=0; k<ZBLOCK_SIZE; k++){
                     for(int j=0; j<YBLOCK_SIZE; j++){
                        #pragma unroll 64
                        for(int i=0; i<XBLOCK_SIZE; i++){
                            volume[(i + blockx) + ((j + blocky)) * XN_MAX + ((k + blockz)) * XN_MAX * YN_MAX] = tuile_voxel[i + (j) * XBLOCK_SIZE + (k) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn; 
                        }
                    }
                }

                 
            }   
        }       
    }
}

__attribute__((scheduler_target_fmax_mhz(250)))
__kernel void backprojection3D_2(
        __global float * restrict volume,
        __global const volatile float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
    
    float __attribute__((numbanks(1)))
    	  __attribute__((doublepump))
    	  __attribute__((numwriteports(1)))
 	      __attribute__((numreadports(64))) cached_sino[taille*tailley];
 	 
 	  
    float __attribute__((numbanks(1)))
    	  //__attribute__((doublepump))
    	  __attribute__((numwriteports(2)))
 	      __attribute__((numreadports(2))) tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
   
    float s_xn_0 = sampling->xn_0; 
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

    
    int temp;

	
    #pragma loop_coalesce 3
    for(uint blockz = 128; blockz < 192; blockz+=ZBLOCK_SIZE){
        for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
            for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
                mean_global=0.0;

		#pragma unroll 64
                for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
                    tuile_voxel[ptr]=0.0f;
                }

		#pragma max_concurrency 4
                for (phi = 0; phi < PHI_MAX; phi++){
                    float xn0, yn0, zn0;
                    xn0 = (float) (blockx - s_xn_0);
                    yn0 = (float) (blocky - s_yn_0);
                    zn0 = (float) (blockz - s_zn_0);
                    float wn0 = 1.0f / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
                    uint un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
                    uint vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
                    
                    uint un0_pr=0;
                    uint vn0_pr=0;
                    
                   // Zone memoire à precharger dependant de un et vn
                    float indice = ((vn0 + tailley/2)/(tailley));
                    
                    vn0_pr = (abs(vn0 - 2))&255; //ramener vn au multiple proche de la hauteur du tableau
                    
                    uint temp = vn0_pr;
                    
                    for(int m=0; m<tailley; m++){
                        un0_pr=0;
                        #pragma unroll 8
                        for(int n=0; n<taille; n++){
                            cached_sino[m*taille +n] = sinogram[(n + (vn0_pr) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                            //un0_pr =(un0_pr + 1);
                            
                        }
                        vn0_pr =(vn0_pr + 1)&(UN_MAX - 1);
                    }
                    
                    for (int zz = 0; zz < ZBLOCK_SIZE; zz++) {
                    	
                        for (int yy = 0; yy < YBLOCK_SIZE; yy++) {
                            
                            
                            #pragma unroll 
                            for (int xx = 0; xx < XBLOCK_SIZE; xx++) { 
                    		
                                xn = xx + blockx;
                                yn = yy + blocky;
                                zn = zz + blockz;
                    
                                 
                                xn_prime = (float) (xn - s_xn_0);
                                yn_prime = (float) (yn - s_yn_0);
                                zn_prime = (float) (zn - s_zn_0);
                                
                                voxel_sum = 0.0f;
                                
		                    float sinogram_coeff = 0;
		                    unsigned short set_un=0, set_vn=0;
		                    
		                    wn_inverse = 1.0f / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
		                    un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
		                    un = un_prime + un_0;
		                    vn_prime = g_vn * zn_prime * wn_inverse;
		                    vn = vn_prime + vn_0;
                                
                                 if ( (un >= 0.0f) && (un < s_d_un - 1.0f) && (vn >= 0.0f) && (vn < s_d_vn - 1.0f) ) {
                                    //un_e = (int) trunc(un);
                                    //vn_e = (int) trunc(vn);
                                    set_un=1;
                                    set_vn=1;
                                    sinogram_coeff = 1; 
                                    
                                    //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
                                }
                                else //un ou vn en dehors des bornes
                                {
                                    
                                    if ((un >= 0.0f)&&(un < s_d_un - 1.0f )) 
                                    {
                                        
                                        //un_e = (int) trunc(un);
                                        set_un=1;

                                        if ( ((vn >= -0.5f) && (vn <= 0.0f)) || ((vn <= s_d_vn - 0.5f) && (vn > 0.0f)) ) {
                                            //vn_e = 0;
                                            set_vn=0;
                                            sinogram_coeff = 1;
                                        }
                                    }
                                    else {
                                        
                                        if ((vn >= 0.0f)&&(vn < s_d_vn - 1.0f )) { 
                                            //vn_e = (int) trunc(vn);
                                            set_vn=1;
                                            
                                            if ( ((un >= -0.5f) && (un <= 0.0f)) || ((un <= s_d_un - 0.5f) && (un > 0.0f)) ) {
                                                //un_e = 0;
                                                set_un=0;
                                                sinogram_coeff = 1;
                                                
                                            }
                                        }
                                    }
                                } 
                                un_e = (unsigned short) (un) * set_un;
                                vn_e = (unsigned short) (vn) * set_vn;
                                
                                
                                uint VN = modulo(vn_e, temp);
                                if(VN ==0 ){
                                    VN= vn_e - temp;
                                }
                                   
                                    //if((vn_e)%(temp) < tailley)
                                    if(VN < tailley)
                                        tuile_voxel[xx + (yy) * XBLOCK_SIZE + (zz) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[((un_e)&(taille)-1) + (VN) * taille]  * sinogram_coeff;
                                    
                                }
  
                                //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
        
                            }
                        } 
                  
                } 
            #pragma loop_coalesce 3
            for(int k=0; k<ZBLOCK_SIZE; k++){
                     for(int j=0; j<YBLOCK_SIZE; j++){
                        #pragma unroll 64
                        for(int i=0; i<XBLOCK_SIZE; i++){
                            volume[(i + blockx) + ((j + blocky)) * XN_MAX + ((k + blockz)) * XN_MAX * YN_MAX] = tuile_voxel[i + (j) * XBLOCK_SIZE + (k) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn; //
                        }
                    }
                }

                 
            }   
        }       
    }
}


__attribute__((scheduler_target_fmax_mhz(250)))
__kernel void backprojection3D_3(
        __global float * restrict volume,
        __global const volatile float * restrict sinogram,
        __constant type_struct_sampling_opencl * restrict sampling,
        __constant float2 * restrict alpha_beta,
        float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
{
    float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
    unsigned int xn, yn, zn, phi;
    unsigned int un_e,vn_e;
    float voxel_sum;
    
    float __attribute__((numbanks(1)))
    	  __attribute__((doublepump))
    	  __attribute__((numwriteports(1)))
 	      __attribute__((numreadports(64))) cached_sino[taille*tailley];
 	 
 	  
    float __attribute__((numbanks(1)))
    	  //__attribute__((doublepump))
    	  __attribute__((numwriteports(2)))
 	      __attribute__((numreadports(2))) tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
   
 
    float s_xn_0 = sampling->xn_0; 
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;

    int temp;

	
    #pragma loop_coalesce 3
    for(uint blockz = 192; blockz < ZN_MAX; blockz+=ZBLOCK_SIZE){
        for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
            for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
                mean_global=0.0;

		#pragma unroll 64
                for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
                    tuile_voxel[ptr]=0.0f;
                }

		#pragma max_concurrency 4
                for (phi = 0; phi < PHI_MAX; phi++){
                    float xn0, yn0, zn0;
                    xn0 = (float) (blockx - s_xn_0);
                    yn0 = (float) (blocky - s_yn_0);
                    zn0 = (float) (blockz - s_zn_0);
                    float wn0 = 1.0f / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
                    uint un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
                    uint vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
                    
                    uint un0_pr=0;
                    uint vn0_pr=0;
                    
                   // Zone memoire à precharger dependant de un et vn
                    float indice = ((vn0 + tailley/2)/(tailley));
                    
                    vn0_pr = (abs(vn0 - 2))&255; //ramener vn au multiple proche de la hauteur du tableau
                    
                    uint temp = vn0_pr;
                    
                    for(int m=0; m<tailley; m++){
                        un0_pr=0;
                        #pragma unroll 8
                        for(int n=0; n<taille; n++){
                            cached_sino[m*taille +n] = sinogram[(n + (vn0_pr) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                            //un0_pr =(un0_pr + 1);
                            
                        }
                        vn0_pr =(vn0_pr + 1)&(UN_MAX - 1);
                    }
                    
                    for (int zz = 0; zz < ZBLOCK_SIZE; zz++) {
                    	
                        for (int yy = 0; yy < YBLOCK_SIZE; yy++) {
                            
                            
                            #pragma unroll 
                            for (int xx = 0; xx < XBLOCK_SIZE; xx++) { 
                    		
                                xn = xx + blockx;
                                yn = yy + blocky;
                                zn = zz + blockz;
                    
                                
                                xn_prime = (float) (xn - s_xn_0);
                                yn_prime = (float) (yn - s_yn_0);
                                zn_prime = (float) (zn - s_zn_0);
                                
                                voxel_sum = 0.0f;
                                
		                    float sinogram_coeff = 0;
		                    unsigned short set_un=0, set_vn=0;
		                    
		                    wn_inverse = 1.0f / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
		                    un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
		                    un = un_prime + un_0;
		                    vn_prime = g_vn * zn_prime * wn_inverse;
		                    vn = vn_prime + vn_0;
                                
                                 if ( (un >= 0.0f) && (un < s_d_un - 1.0f) && (vn >= 0.0f) && (vn < s_d_vn - 1.0f) ) {
                                    //un_e = (int) trunc(un);
                                    //vn_e = (int) trunc(vn);
                                    set_un=1;
                                    set_vn=1;
                                    sinogram_coeff = 1; 
                                    
                                    //voxel_sum += sinogram[un_e + vn_e * UN_MAX + phi * UN_MAX * VN_MAX];
                                }
                                else //un ou vn en dehors des bornes
                                {
                                    
                                    if ((un >= 0.0f)&&(un < s_d_un - 1.0f )) 
                                    {
                                        
                                        //un_e = (int) trunc(un);
                                        set_un=1;

                                        if ( ((vn >= -0.5f) && (vn <= 0.0f)) || ((vn <= s_d_vn - 0.5f) && (vn > 0.0f)) ) {
                                            //vn_e = 0;
                                            set_vn=0;
                                            sinogram_coeff = 1;
                                        }
                                    }
                                    else {
                                        
                                        if ((vn >= 0.0f)&&(vn < s_d_vn - 1.0f )) { 
                                            //vn_e = (int) trunc(vn);
                                            set_vn=1;
                                            
                                            if ( ((un >= -0.5f) && (un <= 0.0f)) || ((un <= s_d_un - 0.5f) && (un > 0.0f)) ) {
                                                //un_e = 0;
                                                set_un=0;
                                                sinogram_coeff = 1;
                                                
                                            }
                                        }
                                    }
                                } 
                                un_e = (unsigned short) (un) * set_un;
                                vn_e = (unsigned short) (vn) * set_vn;
                                
                                
                                uint VN = modulo(vn_e, temp);
                                if(VN ==0 ){
                                    VN= vn_e - temp;
                                }
                                   
                                    //if((vn_e)%(temp) < tailley)
                                    if(VN < tailley)
                                        tuile_voxel[xx + (yy) * XBLOCK_SIZE + (zz) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[((un_e)&(taille)-1) + (VN) * taille]  * sinogram_coeff;
                                    
                                }
  
                                //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
        
                            }
                        } 
                  
                } 
            #pragma loop_coalesce 3
            for(int k=0; k<ZBLOCK_SIZE; k++){
                     for(int j=0; j<YBLOCK_SIZE; j++){
                        #pragma unroll 64
                        for(int i=0; i<XBLOCK_SIZE; i++){
                            volume[(i + blockx) + ((j + blocky)) * XN_MAX + ((k + blockz)) * XN_MAX * YN_MAX] = tuile_voxel[i + (j) * XBLOCK_SIZE + (k) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn; 
                        }
                    }
                }

                 
            }   
        }       
    }
}

