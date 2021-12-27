/*
 * Sinogram3D_GPU_kernel.cu
 *
 *      Author: gac
 */


 #ifndef _SINOGRAM3D_GPU_KERNEL_H_
 #define _SINOGRAM3D_GPU_KERNEL_H_

template <typename T>
__global__ void setSinogram_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = value_r;
}

template <typename T>
__global__ void scalarSinogram_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] *= value_r;
}

template <typename T>
__global__ void addSinogram_k(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id]+b[id];
}

template <typename T>
__global__ void diffSinogram_k(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id]-b[id];
}

template <typename T>
__global__ void diffSinogram_mgpu_k(T *a, T *b, T *c, unsigned long int n)
{

	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;


	id=xn+yn*uSinogramPixelNb_GPU + zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		c[id] = a[id]-b[id];
	}


}

template <typename T>
__global__ void diffSinogram_mgpu_k_debug(T *a, T *b, T *c, unsigned long int n)
{

	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;


	id=xn+yn*uSinogramPixelNb_GPU + zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		//a[id] = 1;
		c[id] = a[id];
	}


}


template <typename T>
__global__ void multSinogram_k(T *a, T *b, T *c, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*uSinogramPixelNb_GPU+zn*uSinogramPixelNb_GPU*vSinogramPixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id]*b[id];
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce(T *g_idata, T *g_odata, unsigned long int n)
{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i+blockSize];

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




template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_square_mgpu(T* g_idata, T* g_odata, unsigned long int n)
{

	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	unsigned int gridSize = blockDim.x*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += (T)(g_idata[i]*g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockDim.x < n)
			mySum += (T)(g_idata[i+blockDim.x]*g_idata[i+blockDim.x]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockDim.x >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockDim.x >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockDim.x >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >=  64)
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







template <class T, unsigned int blockSize,  bool nIsPow2>
__global__ void sinogramReduce_square(T *g_idata, T *g_odata, unsigned long int n)
{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	unsigned int gridSize = blockDim.x*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += (T)(g_idata[i]*g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockDim.x < n)
			mySum += (T)(g_idata[i+blockDim.x]*g_idata[i+blockDim.x]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockDim.x >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockDim.x >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockDim.x >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	__syncthreads();

	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >=  64)
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

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_abspow(T *g_idata, T *g_odata, unsigned long int n, T power)
{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T power_r = power;
	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += pow((T)fabs(g_idata[i]),power_r);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += pow((T)fabs(g_idata[i+blockSize]),power_r);

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

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sinogramReduce_std(T *g_idata, T *g_odata, unsigned long int n, T mean)
{
	T* sdata = SharedMemory<T>();
	T mean_r = mean;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += (mean_r - g_idata[i])*(mean_r - g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += (mean_r - g_idata[i+blockSize])*(mean_r - g_idata[i+blockSize]);

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