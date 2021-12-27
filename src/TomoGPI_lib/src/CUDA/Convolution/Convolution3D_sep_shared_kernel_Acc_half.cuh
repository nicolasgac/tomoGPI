//Date 			:	09/06/2010
//Description 	: 	Convolution 3D Séparable- implémentation GPU // KERNELs
//Auteur 	  	: 	Benoît Pencrec'h
//-------------------------------------------------------------------------------------------------------------------------------


#ifndef _CONVOLUTION3D_SEP_SHARED_KERNEL_ACC_HALF_H_
#define _CONVOLUTION3D_SEP_SHARED_KERNEL_ACC_HALF_H_

////////////////////////////////////////////////FIRST SLICES KERNELS/////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------------------------------------------------------
__global__ 	void  ConvKernel_h_shared_Acc_half	 (unsigned short* d_Output,unsigned short* d_Input)
{
	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[BLOCK_SIZE_H_Y][(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_H_X];

	int volume_offset=__mul24(c_volume_x, c_volume_y);

	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_z		= blockIdx.y/( c_volume_y/blockDim.y);
	const int pixel_y		= __mul24(blockIdx.y%(c_volume_y/blockDim.y), blockDim.y) + threadIdx.y;
	const int pixel_x 	= IMUL( (IMUL(blockIdx.x, NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.x ) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;

	d_Output += pixel_pos ;
	d_Input  += pixel_pos ;

	// Stage 1 : Shared memory loading (for current and next computed slice)-----------------------------------------------------

	// Stage 1.1 : Fill left halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if ( pixel_x >= -pos*BLOCK_SIZE_H_X) //negative sign because we are outside block boudaries
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = __half2float(d_Input[pos*BLOCK_SIZE_H_X]) ;
		}
		else
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = __half2float(0) ;
		}
	}

	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = __half2float(d_Input[pos*BLOCK_SIZE_H_X]) ;
	}

	// Stage 1.3 : Fill right halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if ( c_volume_x - pixel_x > pos*BLOCK_SIZE_H_X )
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = __half2float(d_Input[pos*BLOCK_SIZE_H_X]) ;
		}
		else
		{
			s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X] = __half2float(0) ;
		}
	}

	__syncthreads(); //wait for stage 1 to finish

	// Stage 2 : Row-wise convolution computation -------------------------------------------------------------------------------
	float tmp_sum = 0;

	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;

		for (int k = - (c_kernel_radius_x) ; k <= ((int)(c_kernel_radius_x)); k++)
			tmp_sum += s_Data[threadIdx.y][threadIdx.x + pos*BLOCK_SIZE_H_X + k] * c_Kernel_h[c_kernel_radius_x - k];  

		d_Output[pos*BLOCK_SIZE_H_X] = __float2half_rn(tmp_sum+__half2float(d_Output[pos*BLOCK_SIZE_H_X]));
	}
}

//-------------------------------------------------------------------------------------------------------------------------------
__global__ 	void  ConvKernel_v_shared_Acc_half	 (unsigned short* d_Output,unsigned short* d_Input,int down)
{
	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_V_Y][BLOCK_SIZE_V_X];  

	int volume_offset=__mul24(c_volume_x, c_volume_y);
	int volume_offset2=__mul24(c_volume_x,BLOCK_SIZE_V_Y );

	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_z	= blockIdx.y/(c_volume_y/ (blockDim.y*NUMBER_COMPUTED_BLOCK));	
	const int pixel_y 	= IMUL( (IMUL(blockIdx.y%(c_volume_y/(blockDim.y*NUMBER_COMPUTED_BLOCK)), NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.y ) + threadIdx.y;
	int pixel_x 		= __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;

	d_Output += pixel_pos ;
	d_Input  += down*volume_offset*BLOCK_SIZE_P_Z+pixel_pos ;

	// Stage 1 : Shared memory loading ------------------------------------------------------------------------------------------

	// Stage 1.1 : Fill upper halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if ( pixel_y >= -pos*BLOCK_SIZE_V_Y) //negative sign because we are outside block boudaries
			s_Data[threadIdx.y+ pos*BLOCK_SIZE_V_Y][threadIdx.x ] = __half2float(d_Input[pos*volume_offset2]) ;

		else
			s_Data[threadIdx.y+ pos*BLOCK_SIZE_V_Y][threadIdx.x ] = __half2float(0) ;
	}

	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
		s_Data[threadIdx.y+ pos*BLOCK_SIZE_V_Y][threadIdx.x ] = __half2float(d_Input[pos*volume_offset2]) ;


	// Stage 1.3 : Fill lower halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if ( c_volume_y - pixel_y > pos*BLOCK_SIZE_V_Y )
			s_Data[threadIdx.y+ pos*BLOCK_SIZE_V_Y][threadIdx.x ] = __half2float(d_Input[pos*volume_offset2]) ;

		else
			s_Data[threadIdx.y + pos*BLOCK_SIZE_V_Y][threadIdx.x] = __half2float(0) ;
	}

	__syncthreads(); //wait for stage 1 to finish

	// Stage 2 : Column-wise convolution computation ----------------------------------------------------------------------------
	float tmp_sum = 0;

	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;

		for (int k = - (c_kernel_radius_y) ; k <= ((int)(c_kernel_radius_y));k++ )
			tmp_sum += s_Data[threadIdx.y+ pos*BLOCK_SIZE_V_Y + k][threadIdx.x ] * c_Kernel_v[c_kernel_radius_y - k];

		d_Output[pos*volume_offset2] = __float2half_rn(tmp_sum+__half2float(d_Output[pos*volume_offset2]));
	}
}


//-------------------------------------------------------------------------------------------------------------------------------
__global__ 	void  ConvKernel_p_shared_Acc_half	 (unsigned short* d_Output,unsigned short* d_Input,int down,int up)
{
	// Shared memory declaration, including edge halo zones
	__shared__ float s_Data[(NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK) * BLOCK_SIZE_P_Z][BLOCK_SIZE_P_X];  

	int volume_offset=__mul24(c_volume_x, c_volume_y);
	int volume_offset2=__mul24(BLOCK_SIZE_P_Z, volume_offset);

	// Offset thread position outside the left gird boundary, in order to load zero padded zones of shared memory
	const int pixel_y	= blockIdx.y/(c_volume_z/ (blockDim.y*NUMBER_COMPUTED_BLOCK));
	const int pixel_z 	= IMUL( (IMUL(blockIdx.y%(c_volume_z/(blockDim.y*NUMBER_COMPUTED_BLOCK)), NUMBER_COMPUTED_BLOCK) - NUMBER_HALO_BLOCK ), blockDim.y ) + threadIdx.y;
	int pixel_x 		= __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int pixel_pos = __mul24(pixel_z, volume_offset) + IMUL(pixel_y, c_volume_x) + pixel_x ;

	d_Output += pixel_pos ;
	d_Input  += down*volume_offset2+pixel_pos ;

	// Stage 1 : Shared memory loading ------------------------------------------------------------------------------------------

	// Stage 1.1 : Fill upper halo with data or with zeros if on image border
	for (int pos = 0 ; pos < NUMBER_HALO_BLOCK ; pos++)
	{
		if  (( pixel_z >= -pos*BLOCK_SIZE_P_Z) || ((down ==1))) //negative sign because we are outside block boudaries
			s_Data[threadIdx.y + pos*BLOCK_SIZE_P_Z][threadIdx.x] = __half2float(d_Input[pos*volume_offset2]) ;

		else
			s_Data[threadIdx.y+ pos*BLOCK_SIZE_P_Z][threadIdx.x ] = __half2float(0) ;
	}

	// Stage 1.2 : Fill center data
	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
		s_Data[threadIdx.y + pos*BLOCK_SIZE_P_Z][threadIdx.x] = __half2float(d_Input[pos*volume_offset2]) ;


	// Stage 1.3 : Fill lower halo with data or with zeros if on image border
	for (int pos = NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + 2*NUMBER_HALO_BLOCK ; pos++)
	{
		if (( c_volume_z - pixel_z > pos*BLOCK_SIZE_P_Z ) || ((up == 1)))
			s_Data[threadIdx.y + pos*BLOCK_SIZE_P_Z][threadIdx.x] = __half2float(d_Input[pos*volume_offset2]) ;

		else
			s_Data[threadIdx.y+ pos*BLOCK_SIZE_P_Z][threadIdx.x ] = __half2float(0) ;
	}

	__syncthreads(); //wait for stage 1 to finish


	// Stage 2 : Column-wise convolution computation ----------------------------------------------------------------------------
	float tmp_sum = 0;

	for (int pos = 	NUMBER_HALO_BLOCK ; pos < NUMBER_COMPUTED_BLOCK + NUMBER_HALO_BLOCK ; pos++)
	{
		tmp_sum = 0 ;

		for (int k = - (c_kernel_radius_z) ; k <= ((int)(c_kernel_radius_z)); k++)
			tmp_sum += s_Data[threadIdx.y+ pos*BLOCK_SIZE_P_Z + k][threadIdx.x ] * c_Kernel_p[c_kernel_radius_z - k];

		d_Output[pos*volume_offset2] = __float2half_rn(tmp_sum+__half2float(d_Output[pos*volume_offset2]));
	}
}

//-------------------------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////END OF FILE////////////////////////////////////////////////////////////
#endif