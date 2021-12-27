/*
 * Sinogram3D_GPU_kernel_half.cu
 *
 *      Author: gac
 */


 #ifndef _SINOGRAM3D_GPU_KERNEL_HALF_H_
 #define _SINOGRAM3D_GPU_KERNEL_HALF_H_


template <typename T>
__global__ void setSinogram_k_half(T *a, float value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	float value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
		a_tmp[id] = __float2half_rn(value_r);
}

template <typename T>
__global__ void scalarSinogram_k_half(T *a, float value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	float value_r = value;
	float tmp=0;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id])*value_r;
		a_tmp[id] = __float2half_rn(tmp);
	}

}

template <typename T>
__global__ void addSinogram_k_half(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
	float tmp;

	unsigned short* c_tmp = (unsigned short*)c;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) + __half2float(b[id]);
		c_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void diffSinogram_k_half(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
	float tmp;

	unsigned short* c_tmp = (unsigned short*)c;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) - __half2float(b[id]);
		c_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void multSinogram_k_half(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;
	float tmp;

	unsigned short* c_tmp = (unsigned short*)c;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id])*__half2float(b[id]);
		c_tmp[id] = __float2half_rn(tmp);
	}
}

template <class T, class P, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_half(T *g_idata, P *g_odata, unsigned long int n)
{
	P* sdata = SharedMemory<P>();

//	unsigned short* g_odata_tmp = (unsigned short*)g_odata;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	P mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += __half2float(g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += __half2float(g_idata[i+blockSize]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64)
			mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mySum, offset,32);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = mySum;
}

template <class T, class P, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_square_half(T *g_idata, P *g_odata, unsigned long int n)
{
	P* sdata = SharedMemory<P>();

//	unsigned short* g_odata_tmp = (unsigned short*)g_odata;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	P mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += __half2float(g_idata[i])*__half2float(g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += __half2float(g_idata[i+blockSize])*__half2float(g_idata[i+blockSize]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64)
			mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mySum, offset,32);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = mySum;
}

template <class T, class P, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_abspow_half(T *g_idata, P *g_odata, unsigned long int n, P power)
{
	P* sdata = SharedMemory<P>();

//	unsigned short* g_odata_tmp = (unsigned short*)g_odata;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	P power_r = power;

	P mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += pow((P)fabs(__half2float(g_idata[i])),power_r);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += pow((P)fabs(__half2float(g_idata[i+blockSize])),power_r);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64)
			mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mySum, offset,32);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = mySum;
}

template <class T, class P, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_std_half(T *g_idata, P *g_odata, unsigned long int n, P mean)
{
	P* sdata = SharedMemory<P>();
	P mean_r = mean;

//	unsigned short* g_odata_tmp = (unsigned short*)g_odata;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	P mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += (mean_r - __half2float(g_idata[i]))*(mean_r - __half2float(g_idata[i]));

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += (mean_r - __half2float(g_idata[i+blockSize]))*(mean_r - __half2float(g_idata[i+blockSize]));

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64)
			mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mySum, offset,32);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = mySum;
}

#endif