 /*
 * backprojection3D_kernel.cl
 *
 *      Author: diakite
 */
 
 //#ifndef _BACKPROJECTION3D_KERNEL_CL_
 //#define _BACKPROJECTION3D_KERNEL_CL_

#define PHI_MAX 256
#define XN_MAX 256
#define YN_MAX 256
#define ZN_MAX 256
#define UN_MAX 256
#define VN_MAX 256

#define XBLOCK_SIZE 64
#define YBLOCK_SIZE 64 
#define ZBLOCK_SIZE 4
 
#define taille 256
#define tailley 8

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
    return ((long)x * (long)(y)) >> 32; // Easy way to approximate the modulo function
}

//kernel for DATA REUSE EVALUATION
__kernel void backprojection3D(
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
    //int taille = 128;
    float cached_sino[taille*tailley]={0.0};
    float tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
   // float2 c_ab_wn[PHI_MAX]; // Pour contenir constante alpha et beta
 
    float s_xn_0 = sampling->xn_0; 
    float s_yn_0 = sampling->yn_0;
    float s_zn_0 = sampling->zn_0;
    float s_d_xn = sampling->delta_xn;
    int s_d_un = sampling->N_un;
    int s_d_vn = sampling->N_vn;
    // printf("%f\n", cntTmp);
    // printf("%f\n", s_yn_0);
    // printf("%f\n", s_zn_0);
    // printf("%f\n", s_d_xn);
    // printf("%d\n", s_d_un);
    // printf("%d\n", s_d_vn);
    float mean_global = 0, mean_average = 0; //Data reusefor all phi view
    int n=1; float somme=0.0, mean;
    unsigned int temp;
    float val_min, val_max, val_temp;
  
	//float g_wn = constante->gamma_wn;
	//float un_0 = constante->un_0;
	//float vn_0 = constante->vn_0;
	//float g_vn = constante->gamma_vn;
    //printf("%f\n", sampling->N_un);

    //#pragma unroll PHI_MAX 
    
    //printf("Kernel execution\n");
    //xn = get_global_id(0);
    //yn = get_global_id(1);
    //zn = get_global_id(2);
    for(uint blockz = 0; blockz < ZN_MAX; blockz+=ZBLOCK_SIZE){
        for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
            for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
                mean_global=0.0;

                for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
                    tuile_voxel[ptr]=0.0;
                }
                mean_global = 0;
                for (phi = 0; phi < PHI_MAX; phi++){
                    float xn0, yn0, zn0;
                    xn0 = (float) (blockx - s_xn_0);
                    yn0 = (float) (blocky - s_yn_0);
                    zn0 = (float) (blockz - s_zn_0);
                    float wn0 = 1.0 / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
                    int un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
                    int vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
                    if(un0<0) un0 = 0;
                    if(vn0<0) vn0 = 0;
                    //printf("%d %d\t", un0, vn0);
                    int un0_pr=0;
                    int vn0_pr=0;
                    //vn0_pr = abs(vn0);
                   // Zone memoire à precharger dependant de un et vn
                   // float indice = ((vn0 + tailley/2)/(tailley));
                    // if(vn0>8){
                    // vn0_pr = floor(indice)*tailley; //ramener vn au multiple proche de la hauteur du tableau
                    // vn0_pr = vn0_pr;
                    // }
                    //if(vn0>3) 
                    vn0_pr = abs(vn0 - 2); //ramener vn au multiple proche de la hauteur du tableau
                    //printf("vn0: %d vn_pr: %d\t", vn0, vn0_pr);
                    uint temp = vn0_pr;
                    //printf("%d\t", temp);
                    // for(int m=0; m<tailley; m++){
                    //     un0_pr=0;
                    //     for(int n=0; n<taille; n++){
                    //         cached_sino[m*taille +n] = sinogram[((n)%256 + (vn0_pr%256) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                    //         un0_pr =(un0_pr + 1);
                    //         //cached_sino[m*XBLOCK_SIZE +n] = sinogram[(un0_pr%256 + (vn0_pr%256) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
                    //         //un0_pr =(un0_pr + 1);
                    //         /*if(un0_pr >= UN_MAX){
                    //             un0_pr = un0;
                    //             vn0_pr =(vn0_pr + 1)%UN_MAX; 
                    //         }
                    //         else{
                    //             un0_pr =(un0_pr + 1);
                    //         }*/
                    //     }
                    //     vn0_pr =(vn0_pr + 1)%UN_MAX;
                    // }
                    //int indice=0;
                    for (zn = blockz; zn < blockz + ZBLOCK_SIZE; zn++) {
                        for (yn = blocky; yn < blocky + YBLOCK_SIZE; yn++){
                            
                            //#pragma max_concurrency XN_MAX
                            for (xn = blockx; xn < blockx + XBLOCK_SIZE; xn++) {  
                    
                                //printf("%d\n", xn); 
                                xn_prime = (float) (xn - s_xn_0);
                                yn_prime = (float) (yn - s_yn_0);
                                zn_prime = (float) (zn - s_zn_0);
                                //printf("%f\n", s_d_xn);
                                // xn_prime = (float) (xn0 + xn%XBLOCK_SIZE );
                                // yn_prime = (float) (yn0 + yn%XBLOCK_SIZE );
                                // zn_prime = (float) (zn0 + zn%ZBLOCK_SIZE );
 
                                //int i=blockx + xn;
                                //int j=blocky + yn;
                                //int k=blockz + zn;
                                // float voxel_sum;
                                voxel_sum = 0.0f;
                                //barrier(CLK_LOCAL_MEM_FENCE);
                                //float voxel_sum_tab[PHI_MAX]; // partition en 4 
 
                                //#pragma unroll 32
                                //for (phi = 0; phi < PHI_MAX; phi++){
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
                                    // if(vn0 < 0){
                                    //     printf("%d != %d\t", vn_e, vn0); 
                                    // }
                                    //voxel_sum += sinogram[un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX] * sinogram_coeff;
                                    //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
                                    //volume[xn%256 + (yn%256) * XN_MAX + (zn%256) * XN_MAX * YN_MAX] += cached_sino[((un_e - un0)%XBLOCK_SIZE) + ((vn_e - vn0)%ZBLOCK_SIZE) * XBLOCK_SIZE] * s_d_xn * sinogram_coeff * 0.00390625;
                                    //if(temp > 12)
                                    //printf("%d\t", zn);
                                    uint VN = modulo(vn_e, temp);
                                    //if((vn_e)%(temp) != VN){
                                    if(VN ==0 ){
                                        VN=(vn_e)-(temp);
                                        // if((vn_e)==(temp))
                                        // printf("%u : %u\t", vn_e, temp);
                                    }

                                    if(VN < tailley){
                                    //if((vn_e)%(temp) < tailley){
                                        //printf("%d\t", (vn_e)%(temp));
                                        //volume[xn%256 + (yn%256) * XN_MAX + (zn%256) * XN_MAX * YN_MAX] += cached_sino[((un_e)%taille) + ((vn_e)%(temp)) * taille]  * sinogram_coeff * s_d_xn;
                                        //tuile_voxel[xn%XBLOCK_SIZE + (yn%YBLOCK_SIZE) * XBLOCK_SIZE + (zn%ZBLOCK_SIZE) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[(((un_e)%taille) + (VN) * taille)]  * sinogram_coeff;
                                        //volume[i + j * XN_MAX + k * XN_MAX * YN_MAX] += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
                                        //volume[un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX] +=1;
                                        cached_sino[(((un_e)%taille) + (VN) * taille)]+=1; //data reuse calculation
                                    //     if((vn_e)%(temp) != VN){
                                    //         if((vn_e)%(temp) > 5)
                                    //             printf("%d : %d\t", (vn_e)%(temp), VN);
                                    //     }
                                    }
                                }
                            //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
                            }
                        } 
                    somme = 0;
                    n = 1;
                    for(int l=0; l<tailley; l++){
                        for(int p=0; p<256; p++){
                            if(cached_sino[p + l*256]!=0.0){
                                somme += cached_sino[p + l*256];
                                n++;
                            }
                            cached_sino[p + l*256] = 0.0;

                        }
                    }
                    //mean=somme/n;
                    mean=(XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE)/n;
                    if(phi==0){
                        val_min = mean;
                        val_max = mean;
                    }
                        

                    val_min = min(mean, val_min);
                    val_max = max(mean, val_max);
                    if(mean < 1024)
                        mean_global += mean;

                    //printf("mean = %.1f\t", mean);

                } 
                //mean_global = (val_max + val_min)/2;
                mean_global = mean_global/256;
                // //if(mean_global>230)
                //printf("mean = %.1f\n", mean_global);
                mean_average += mean_global;

                //uint adresse = 0;
                for(uint k=blockz; k<blockz+ZBLOCK_SIZE; k++){
                    for(uint j=blocky; j<blocky+YBLOCK_SIZE; j++){
                        for(uint i=blockx; i<blockx+XBLOCK_SIZE; i++){
                            //if(k<200)
                            volume[i%256 + (j%256) * XN_MAX + (k%256) * XN_MAX * YN_MAX] = tuile_voxel[i%XBLOCK_SIZE + (j%YBLOCK_SIZE) * XBLOCK_SIZE + (k%ZBLOCK_SIZE) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn;
                            //adresse++;
                        }
                    }
                }   
 
                 
            }   
        }       
    }
    // somme = 0;
    // for(int k=0; k<256; k++){
    //     for(int l=0; l<256; l++){
    //         for(int p=0; p<256; p++){
    //             if(volume[p + l*256 + k*256*256]!=0.0){
    //                 somme += volume[p + l*256 + k*256*256];
    //                 n++;
    //             }
    //         }
    //     }
    // }
    // mean_global = somme/n;
    printf("mean_total = %.1f\n", mean_average/((XN_MAX*YN_MAX*ZN_MAX)/(XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE)));
}



// __kernel void backprojection3D_bloc(
//         __global float * restrict volume,
//         __global float * restrict sinogram,
//         __constant type_struct_sampling_opencl * restrict sampling,
//        // __constant type_struct_constante_opencl * restrict constante,
//         __constant float2 * restrict alpha_beta,
//         float cntTmp)    // cntTmp = (constante->D/constante->delta_un)*sampling->delta_xn
// {
//     float xn_prime, yn_prime, zn_prime, wn_inverse, un_prime, vn_prime, un, vn, epsilon_un, epsilon_vn;
//     unsigned int xn, yn, zn, phi;
//     unsigned int un_e,vn_e;
//     float voxel_sum;
//     //int taille = 128;
//     float cached_sino[taille*tailley];
//     float tuile_voxel[XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE];
//    // float2 c_ab_wn[PHI_MAX]; // Pour contenir constante alpha et beta
 
//     float s_xn_0 = sampling->xn_0; 
//     float s_yn_0 = sampling->yn_0;
//     float s_zn_0 = sampling->zn_0;
//     float s_d_xn = sampling->delta_xn;
//     int s_d_un = sampling->N_un;
//     int s_d_vn = sampling->N_vn;

//     float mean_global; //Data reusefor all phi view
//     unsigned int temp;
  
// 	//float g_wn = constante->gamma_wn;
// 	//float un_0 = constante->un_0;
// 	//float vn_0 = constante->vn_0;
// 	//float g_vn = constante->gamma_vn;
//     //printf("%f\n", sampling->N_un);

//     //#pragma unroll PHI_MAX 
    
//     //printf("Kernel execution\n");
//     //xn = get_global_id(0);
//     //yn = get_global_id(1);
//     //zn = get_global_id(2);
//     for(uint blockz = 0; blockz < ZN_MAX; blockz+=ZBLOCK_SIZE){
//         for(uint blocky = 0; blocky < YN_MAX; blocky+=YBLOCK_SIZE){
//             for(uint blockx = 0; blockx < XN_MAX; blockx+=XBLOCK_SIZE){
//                 mean_global=0.0;

//                 for(int ptr=0; ptr<XBLOCK_SIZE*YBLOCK_SIZE*ZBLOCK_SIZE; ptr++){
//                     tuile_voxel[ptr]=0.0;
//                 }

//                 for (phi = 0; phi < PHI_MAX; phi++){
//                     float xn0, yn0, zn0;
//                     xn0 = (float) (blockx - s_xn_0);
//                     yn0 = (float) (blocky - s_yn_0);
//                     zn0 = (float) (blockz - s_zn_0);
//                     float wn0 = 1.0 / ( (alpha_beta[phi].s0 * xn0 + alpha_beta[phi].s1 * yn0) * s_d_xn + g_wn);
//                     int un0 = (uint) ((yn0 * alpha_beta[phi].s0 - xn0 * alpha_beta[phi].s1) * cntTmp * wn0 + un_0);
//                     int vn0 = (uint) (g_vn * zn0 * wn0 + vn_0);
//                     if(un0<0) un0 = 0;
//                     if(vn0<0) vn0 = 0;
//                     //printf("%d %d\t", un0, vn0);
//                     int un0_pr=0;
//                     int vn0_pr=0;
                    
//                     vn0_pr = abs(vn0 - 2); 
//                     uint temp = vn0_pr;
//                     //printf("%d\t", temp);
//                     for(int m=0; m<tailley; m++){
//                         un0_pr=0;
//                         for(int n=0; n<taille; n++){
//                             cached_sino[m*taille +n] = sinogram[((n)%256 + (vn0_pr%256) * UN_MAX + phi * UN_MAX * VN_MAX)]; 
//                             un0_pr =(un0_pr + 1);
                            
//                         }
//                         vn0_pr =(vn0_pr + 1)%UN_MAX;
//                     }
//                     //int indice=0;
//                     for (zn = blockz; zn < blockz + ZBLOCK_SIZE; zn++) {
//                         for (yn = blocky; yn < blocky + YBLOCK_SIZE; yn++){
                            
//                             //#pragma max_concurrency XN_MAX
//                             for (xn = blockx; xn < blockx + XBLOCK_SIZE; xn++) {  
                    
//                                 //printf("%d\n", xn); 
//                                 xn_prime = (float) (xn - s_xn_0);
//                                 yn_prime = (float) (yn - s_yn_0);
//                                 zn_prime = (float) (zn - s_zn_0);
                                
//                                 voxel_sum = 0.0f;
                                
//                                     float sinogram_coeff = 0;
//                                     wn_inverse = 1.0 / ( (alpha_beta[phi].s0 * xn_prime + alpha_beta[phi].s1 * yn_prime) * s_d_xn + g_wn);
//                                     un_prime = (yn_prime * alpha_beta[phi].s0 - xn_prime * alpha_beta[phi].s1) * cntTmp * wn_inverse;
//                                     un = un_prime + un_0;
//                                     vn_prime = g_vn * zn_prime * wn_inverse;
//                                     vn = vn_prime + vn_0;
                                
//                                     if ( (un >= 0) && (un < s_d_un - 1) && (vn >= 0) && (vn < s_d_vn - 1) ) {
//                                         un_e = (int) trunc(un);
//                                         vn_e = (int) trunc(vn);
//                                         sinogram_coeff = 1; 
                                        
//                                     }
//                                     else //un ou vn en dehors des bornes
//                                     {
                                        
//                                         if ((un >= 0)&&(un < s_d_un - 1 )) 
//                                         {
                                            
//                                             un_e = (int) trunc(un);

//                                             if ( ((vn >= -0.5) && (vn <= 0)) || ((vn <= s_d_vn - 0.5) && (vn > 0)) ) {
//                                                 vn_e = 0;
//                                                 sinogram_coeff = 1;
//                                             }
//                                         }
//                                         else {
                                            
//                                             if ((vn >= 0)&&(vn < s_d_vn - 1 )) { 
//                                                 vn_e = (int) trunc(vn);
                                                
//                                                 if ( ((un >= -0.5) && (un <= 0)) || ((un <= s_d_un - 0.5) && (un > 0)) ) {
//                                                     un_e = 0;
//                                                     sinogram_coeff = 1;
                                                    
//                                                 } 
//                                             }
//                                         }
//                                     } 
                                    
//                                     uint VN = modulo(vn_e, temp);
                                    
//                                     if(VN ==0 ){
//                                         VN=(vn_e)-(temp);
//                                     }

//                                     if(VN < tailley){
//                                         tuile_voxel[xn%XBLOCK_SIZE + (yn%YBLOCK_SIZE) * XBLOCK_SIZE + (zn%ZBLOCK_SIZE) * XBLOCK_SIZE * YBLOCK_SIZE] += cached_sino[(((un_e)%taille) + (VN) * taille)]  * sinogram_coeff;
//                                     }
//                                 }
//                             //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
//                             }
//                         } 
            

//                 } 
            
//             for(uint k=blockz; k<blockz+ZBLOCK_SIZE; k++){
//                 for(uint j=blocky; j<blocky+YBLOCK_SIZE; j++){
//                     for(uint i=blockx; i<blockx+XBLOCK_SIZE; i++){
//                         //if(k<200)
//                         volume[i%256 + (j%256) * XN_MAX + (k%256) * XN_MAX * YN_MAX] = tuile_voxel[i%XBLOCK_SIZE + (j%YBLOCK_SIZE) * XBLOCK_SIZE + (k%ZBLOCK_SIZE) * XBLOCK_SIZE * YBLOCK_SIZE] * s_d_xn;
//                         //adresse++;
//                     }
//                 }
//             }   
 
                 
//             }   
//         }       
//     }
// }





/*
__kernel void backprojection3D(
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
    phi = get_global_id(0);
    yn = get_global_id(1);
    zn = get_global_id(2);

    __local float fecth_data[24*8]; 

    float wn0 = 1.0 / ( (alpha_beta[0].s0 * xn_prime + alpha_beta[0].s1 * yn_prime) * s_d_xn + g_wn);
    float un_prime0 = (yn_prime * alpha_beta[0].s0 - xn_prime * alpha_beta[0].s1) * cntTmp * wn0;
    float u0 = un_prime0 + un_0;
    float vn_prime0 = g_vn * zn_prime * wn0;
    float v0 = vn_prime0 + vn_0;

    fecth(sinogram, fecth_data, u0, v0, phi);

   for (xn = 0; xn < XN_MAX; xn++) {
    xn_prime = (float) (xn - s_xn_0);
    yn_prime = (float) (yn - s_yn_0);
    zn_prime = (float) (zn - s_zn_0);
    //printf("%f\n", s_d_xn);

    // float voxel_sum;
    voxel_sum = 0.0f;
    //barrier(CLK_LOCAL_MEM_FENCE);
    

    //#pragma unroll 32
    
    //for (phi = 0; phi < PHI_MAX; phi++){
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
        int u_local = un_e - u0;
        int v_local = vn_e - v0;
            
        //voxel_sum += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)%256] * sinogram_coeff;
        //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
        volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] += fecth_data[v_local * 8 + u_local] * s_d_xn * sinogram_coeff;
    
    }

    //volume[xn + yn * XN_MAX + zn * XN_MAX * YN_MAX] = voxel_sum * s_d_xn;
    
           
}
*/
//#endif  

 


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
            
        voxel_sum += sinogram[(un_e%256 + (vn_e%256) * UN_MAX + phi * UN_MAX * VN_MAX)] * s_d_xn * sinogram_coeff;
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

                    voxel_sum += sinogram[(un_e & 255) + (vn_e & 255) * UN_MAX + phi * UN_MAX * VN_MAX] * sinogram_coeff;
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






                    // if(vn0 > (YN_MAX-32*8))
                    //     vn0_pr=abs(YN_MAX-32*8);

                    // if(vn0 > (YN_MAX-31*8))
                    //     vn0_pr=abs(YN_MAX-31*8);

                    // if(vn0 > (YN_MAX-30*8))
                    //     vn0_pr=abs(YN_MAX-30*8);

                    // if(vn0 > (YN_MAX-29*8))
                    //     vn0_pr=abs(YN_MAX-29*8);

                    // if(vn0 > (YN_MAX-28*8))
                    //     vn0_pr=abs(YN_MAX-28*8);

                    // if(vn0 > (YN_MAX-27*8))
                    //     vn0_pr=abs(YN_MAX-27*8);
                    
                    // if(vn0 > (YN_MAX-26*8))
                    //     vn0_pr=abs(YN_MAX-26*8);

                    // if(vn0 > (YN_MAX-25*8))
                    //     vn0_pr=abs(YN_MAX-25*8);

                    // if(vn0 > (YN_MAX-24*8))
                    //     vn0_pr=(YN_MAX-24*8);

                    // if(vn0 > (YN_MAX-23*8))
                    //     vn0_pr=(YN_MAX-23*8);

                    // if(vn0 > (YN_MAX-22*8))
                    //     vn0_pr=(YN_MAX-22*8);

                    //  if(vn0 > (YN_MAX-21*8))
                    // vn0_pr=abs(YN_MAX-21*8);

                    // if(vn0 > (YN_MAX-20*8))
                    //     vn0_pr=abs(YN_MAX-20*8);

                    // if(vn0 > (YN_MAX-19*8))
                    //     vn0_pr=abs(YN_MAX-19*8);
                    
                    // if(vn0 > (YN_MAX-18*8))
                    //     vn0_pr=(YN_MAX-18*8);

                    // if(vn0 > (YN_MAX-17*8))
                    //     vn0_pr=(YN_MAX-17*8);
                    
                    // if(vn0 > (YN_MAX-16*8))
                    //     vn0_pr=(YN_MAX-16*8);
                    
                    // if(vn0 > (YN_MAX-15*8))
                    //     vn0_pr=abs(YN_MAX-15*8);

                    // if(vn0 > (YN_MAX-14*8))
                    //     vn0_pr=abs(YN_MAX-14*8);

                    // if(vn0 > (YN_MAX-13*8))
                    //     vn0_pr=abs(YN_MAX-13*8);

                    // if(vn0 > (YN_MAX-12*8))
                    //     vn0_pr=abs(YN_MAX-12*8);
                    
                    // if(vn0 > (YN_MAX-11*8))
                    //     vn0_pr=abs(YN_MAX-11*8);

                    // if(vn0 > (YN_MAX-10*8))
                    //     vn0_pr=abs(YN_MAX-10*8);

                    // if(vn0 > (YN_MAX-9*8))
                    //     vn0_pr=(YN_MAX-9*8);

                    // if(vn0 > (YN_MAX-8*8))
                    //     vn0_pr=(YN_MAX-8*8);

                    // if(vn0 > (YN_MAX-7*8))
                    //     vn0_pr=(YN_MAX-7*8);

                    //  if(vn0 > (YN_MAX-6*8))
                    // vn0_pr=abs(YN_MAX-6*8);

                    // if(vn0 > (YN_MAX-5*8))
                    //     vn0_pr=abs(YN_MAX-5*8);

                    // if(vn0 > (YN_MAX-4*8))
                    //     vn0_pr=abs(YN_MAX-4*8);
                    
                    // if(vn0 > (YN_MAX-3*8))
                    //     vn0_pr=(YN_MAX-3*8);

                    // if(vn0 > (YN_MAX-2*8))
                    //     vn0_pr=(YN_MAX-2*8);
                    
                    // if(vn0 > (YN_MAX-8))
                    //     vn0_pr=(YN_MAX-8);