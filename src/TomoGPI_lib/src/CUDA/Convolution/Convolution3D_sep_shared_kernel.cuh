//Date 			:	09/06/2010
//Description 	: 	Convolution 3D Séparable- implémentation GPU // KERNELs
//Auteur 	  	: 	Benoît Pencrec'h
//-------------------------------------------------------------------------------------------------------------------------------


#ifndef _CONVOLUTION3D_SEP_SHARED_KERNEL_H_
#define _CONVOLUTION3D_SEP_SHARED_KERNEL_H_

////////////////////////////////////////////////FIRST SLICES KERNELS/////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------------------------------------------------------
template <typename T>
__global__ 	void  ConvKernel_h_shared	 (T* d_Output,T* d_Input,int down)
{
	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[BLOCK_SIZE_H_Y][(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_H_X];

	int volume_offset=__mul24(c_volume_x, c_volume_y);

	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_z		= blockIdx.y/(c_volume_y/blockDim.y);
	const int pixel_y		= __mul24(blockIdx.y%(c_volume_y/blockDim.y), blockDim.y) + threadIdx.y;
	const int pixel_x 	= IMUL( (IMUL(blockIdx.x, NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.x ) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;

	d_Output += pixel_pos ;
	d_Input  += down*volume_offset*BLOCK_SIZE_P_Z+pixel_pos ;


	// Stage 1 : Shared memory loading (for current and next computed slice)-----------------------------------------------------

	// Stage 1.1 : Fill left halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if ( pixel_x >= -pos*BLOCK_SIZE_H_X) //negative sign because we are outside block boudaries
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = d_Input[pos*BLOCK_SIZE_H_X] ;
		}
		else
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = 0 ;
		}
	}


	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = d_Input[pos*BLOCK_SIZE_H_X] ;
	}


	// Stage 1.3 : Fill right halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if ( c_volume_x - pixel_x > pos*BLOCK_SIZE_H_X )
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = d_Input[pos*BLOCK_SIZE_H_X] ;
		}
		else
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = 0 ;
		}
	}


	__syncthreads(); //wait for stage 1 to finish


	// Stage 2 : Row-wise convolution computation -------------------------------------------------------------------------------
	float tmp_sum = 0;






#if UNROLL_CONV3D
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos+=8)
	{
		tmp_sum = 0 ;
		int a,b,c,d;
		//1
		d=pos*BLOCK_SIZE_H_X;
		a=threadIdx.x + d-c_kernel_radius_x;
		c=a;
		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}

		d_Output[d] = tmp_sum;	
		//2
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	
		//3
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	

		//4
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	

		//5
		a+=BLOCK_SIZE_H_X;
		tmp_sum = 0 ;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	

		//6
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	

		//7
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	

		//8
		tmp_sum = 0 ;
		a+=BLOCK_SIZE_H_X;
		d+=BLOCK_SIZE_H_X;
		c=a;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][c] * c_Kernel_h[b];  
			c++;
		}
		d_Output[d] = tmp_sum;	








	}

#else
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;
		int a,b;

		a=threadIdx.x + pos*BLOCK_SIZE_H_X-c_kernel_radius_x;

		for ( b = 2*c_kernel_radius_x  ; b >= 0; b--){
			tmp_sum += s_Data[threadIdx.y][a] * c_Kernel_h[b];  
			a++;
		}

		d_Output[pos*BLOCK_SIZE_H_X] = tmp_sum;
	}
#endif






}





//-------------------------------------------------------------------------------------------------------------------------------
template <typename T>
__global__ 	void  ConvKernel_v_shared	 (T* d_Output,T* d_Input)
{
	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[BLOCK_SIZE_V_X][(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_V_Y];  

	int volume_offset=__mul24(c_volume_x, c_volume_y);
	int volume_offset2=__mul24(c_volume_x,BLOCK_SIZE_V_Y );

	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_z		= blockIdx.y/( c_volume_x/blockDim.y);	
	const int pixel_x	= IMUL(blockIdx.y%(c_volume_x/blockDim.y), blockDim.y) + threadIdx.y;
	const int pixel_y 	= IMUL( (IMUL(blockIdx.x, NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.x ) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;

	d_Output += pixel_pos ;
	d_Input  += pixel_pos ;


	// Stage 1 : Shared memory loading ------------------------------------------------------------------------------------------

	// Stage 1.1 : Fill upper halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if ( pixel_y >= -pos*BLOCK_SIZE_V_Y) //negative sign because we are outside block boudaries
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y] = d_Input[pos*volume_offset2] ;

		else
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y] = 0 ;
	}


	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
		s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y] = d_Input[pos*volume_offset2] ;


	// Stage 1.3 : Fill lower halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if ( c_volume_y - pixel_y > pos*BLOCK_SIZE_V_Y )
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y] = d_Input[pos*volume_offset2] ;

		else
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y] = 0 ;
	}


	__syncthreads(); //wait for stage 1 to finish


	// Stage 2 : Column-wise convolution computation ----------------------------------------------------------------------------
	float tmp_sum = 0;

	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;

		for (int k = - (c_kernel_radius_y) ; k <= ((int)(c_kernel_radius_y)); k++)
			tmp_sum += s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_V_Y + k] * c_Kernel_v[c_kernel_radius_y - k]	;  

		d_Output[pos*volume_offset2] = tmp_sum;	
	}
}



//-------------------------------------------------------------------------------------------------------------------------------
template <typename T>
__global__ 	void  ConvKernel_p_shared	 (T* d_Output,T* d_Input)
{



	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[BLOCK_SIZE_P_X][(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_P_Z];  

	int volume_offset=__mul24(c_volume_x, c_volume_y);
	int volume_offset2=__mul24(BLOCK_SIZE_P_Z, volume_offset);


	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_y		= blockIdx.y/( c_volume_x/blockDim.y);	
	const int pixel_x	= IMUL(blockIdx.y% (c_volume_x/blockDim.y), blockDim.y) + threadIdx.y;
	const int pixel_z 	= IMUL( (IMUL(blockIdx.x, NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.x ) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;



	d_Output += pixel_pos ;
	d_Input  += pixel_pos ;


	// Stage 1 : Shared memory loading ------------------------------------------------------------------------------------------

	// Stage 1.1 : Fill upper halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if ( pixel_z >= -pos*BLOCK_SIZE_P_Z) //negative sign because we are outside block boudaries
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z] = d_Input[pos*volume_offset2] ;

		else
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z] = 0 ;
	}


	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
		s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z] = d_Input[pos*volume_offset2] ;


	// Stage 1.3 : Fill lower halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if ( c_volume_z - pixel_z > pos*BLOCK_SIZE_P_Z )
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z] = d_Input[pos*volume_offset2] ;

		else
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z] = 0 ;
	}


	__syncthreads(); //wait for stage 1 to finish


	// Stage 2 : Column-wise convolution computation ----------------------------------------------------------------------------
	float tmp_sum = 0;

	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;

		for (int k = - (c_kernel_radius_z) ; k <= ((int)(c_kernel_radius_z)); k++){
			tmp_sum += s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_P_Z + k] * c_Kernel_p[c_kernel_radius_z - k]	;  
			//if ( pixel_z >= -pos*BLOCK_SIZE_P_Z)
			//		tmp_sum += d_Input[(pos*BLOCK_SIZE_P_Z+k)*volume_offset]* c_Kernel_p[c_kernel_radius_z - k]	;
		}


		d_Output[pos*volume_offset2] = tmp_sum;	
		//d_Output[pos*volume_offset2] = pos*volume_offset2 + pixel_pos;	
	}



}

//-------------------------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////END OF FILE////////////////////////////////////////////////////////////
#endif