/*
 * Image3D_GPU_kernel.cu
 *
 *      Author: gac
 */

 #ifndef _IMAGE3D_GPU_KERNEL_H_
 #define _IMAGE3D_GPU_KERNEL_H_

#define sign(a) ((a) > 0 ? +1 : ((a) < 0 ? -1 : 0))

template <typename T>
__global__ void setImage_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n){
		a[id] = value_r;
	}
}



template <typename T>
__global__ void setImage_mgpu_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		a[id] = value_r;
	}
}



template <typename T>
__global__ void scalarImage_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] *= value_r;
}


template <typename T>
__global__ void scalarImage_mgpu_k(T *a, T value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		a[id] *= value_r;
	}
}



template <typename T>
__global__ void addImage_k(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = a[id]+b[id];
}

template <typename T>
__global__ void addImage_k(T *a, T *b, unsigned long int n, T lambda)
{
	unsigned int xn,yn,zn,id;
	T lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = a[id]+lambda_r*b[id];
}

template <typename T>
__global__ void addImage_k_debug(T *a, T *b, unsigned long int n, T lambda)
{
	unsigned int xn,yn,zn,id;
	T lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		b[id] = 5;		//sender
		a[id] = 2;		//receiver
	}

}


template <typename T>
__global__ void positiveAddImage_k(T *a, T *b, unsigned long int n, T lambda)
{
	unsigned int xn,yn,zn,id;
	T lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	T tmp;
	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = a[id] + lambda_r*b[id];
		if(tmp<0)
			a[id] = 0.0;
		else
			a[id] = tmp;
	}
}

template <typename T>
__global__ void diffImage_k(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = a[id]-b[id];
}

template <typename T>
__global__ void diffImage_k(T lambda, T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = lambda_r*a[id] - b[id];
}

template <typename T>
__global__ void diffImage_k_moins_lambda(T lambda, T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	T lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = a[id] - lambda_r*b[id];
}

template <typename T>
__global__ void multImage_k(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		a[id] = a[id]*b[id];
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce(T *g_idata, T *g_odata, unsigned long int n)
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
__global__ void imageReduce_square(T *g_idata, T *g_odata, unsigned long int n)
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
		mySum += g_idata[i]*g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i+blockSize]*g_idata[i+blockSize];

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
__global__ void imageReduce_scalarProduct(T *g_idata1, T *g_idata2, T *g_odata, unsigned long int n)
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
		mySum += g_idata1[i]*g_idata2[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata1[i+blockSize]*g_idata2[i+blockSize];

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
__global__ void imageReduce_abspow(T *g_idata, T *g_odata, unsigned long int n, T power)
{
	T* sdata = SharedMemory<T>();
	T power_r = power;

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
__global__ void imageReduce_huber(T *g_idata, T *g_odata, unsigned long int n, T threshold)
{
	T* sdata = SharedMemory<T>();
	T threshold_r = threshold;
	T squareThreshold_r = threshold_r*threshold_r;
	T tmp = 0;

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
		tmp = g_idata[i];
		if(fabs(tmp) < threshold_r)
			mySum += tmp*tmp;
		else if (fabs(tmp) >= threshold)
			mySum += 2.0*threshold*fabs(tmp)-squareThreshold_r;
		else
			mySum += tmp;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
		{
			if(fabs(tmp) < threshold_r)
				mySum += tmp*tmp;
			else if (fabs(tmp) >= threshold)
				mySum += 2.0*threshold*fabs(tmp)-squareThreshold_r;
			else
				mySum += tmp;
		}

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
__global__ void imageReduce_std(T *g_idata, T *g_odata, unsigned long int n, T mean)
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

template <typename T>
__global__ void signedImage_k(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		b[id] = sign(a[id]);
}

template <typename T>
__global__ void absPowImage_k(T *a, T *b, unsigned long int n,T power)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	T power_r = power;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	// Make sure we do not go out of bounds
	if (id < n)
		b[id] = pow((T)fabs(a[id]),power_r);
}



#endif
