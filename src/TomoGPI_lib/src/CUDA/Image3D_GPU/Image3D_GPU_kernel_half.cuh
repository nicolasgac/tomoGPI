/*
 * Image3D_GPU_kernel_half.cu
 *
 *      Author: gac
 */

 #ifndef _IMAGE3D_GPU_KERNEL_HALF_H_
 #define _IMAGE3D_GPU_KERNEL_HALF_H_

#define sign(a) ((a) > 0 ? +1 : ((a) < 0 ? -1 : 0))

template <typename T>
__global__ void setImage_k_half(T *a, float value, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	float value_r = value;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
		a_tmp[id] = __float2half_rn(value_r);
}

template <typename T>
__global__ void scalarImage_k_half(T *a, float value, unsigned long int n)
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
__global__ void addImage_k_half(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) + __half2float(b[id]);
		a_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void addImage_k_half(T *a, T *b, unsigned long int n, float lambda)
{
	unsigned int xn,yn,zn,id;
	float lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) + lambda_r*__half2float(b[id]);
		a_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void positiveAddImage_k_half(T *a, T *b, unsigned long int n, float lambda)
{
	unsigned int xn,yn,zn,id;
	float lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) + lambda_r*__half2float(b[id]);
		if(tmp<0)
			a_tmp[id] = 0.0;
		else
			a_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void diffImage_k_half(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]) - __half2float(b[id]);
		a_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void diffImage_k_half(float lambda, T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	float lambda_r = lambda;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = lambda_r*__half2float(a[id]) - __half2float(b[id]);
		a_tmp[id] = __float2half_rn(tmp);
	}
}

template <typename T>
__global__ void multImage_k_half(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;
	float tmp;

	unsigned short* a_tmp = (unsigned short*)a;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id])*__half2float(b[id]);
		a_tmp[id] = __float2half_rn(tmp);
	}
}

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_half(T *g_idata, P *g_odata, unsigned long int n)
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

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_square_half(T *g_idata, P *g_odata, unsigned long int n)
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

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_scalarProduct_half(T *g_idata1, T *g_idata2, P *g_odata, unsigned long int n)
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
		mySum += __half2float(g_idata1[i])*__half2float(g_idata2[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += __half2float(g_idata1[i+blockSize])*__half2float(g_idata2[i+blockSize]);

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

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_abspow_half(T *g_idata, P *g_odata, unsigned long int n, P power)
{
	P* sdata = SharedMemory<P>();

	P power_r = power;

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

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_huber_half(T *g_idata, P *g_odata, unsigned long int n, P threshold)
{
	P* sdata = SharedMemory<P>();
	P threshold_r = threshold;
	P squareThreshold_r = threshold_r*threshold_r;
	P tmp = 0;

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
		tmp = __half2float(g_idata[i]);
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

template <class T,class P, unsigned int blockSize, bool nIsPow2>
__global__ void imageReduce_std_half(T *g_idata, P *g_odata, unsigned long int n, P mean)
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

template <typename T>
__global__ void signedImage_k_half(T *a, T *b, unsigned long int n)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	unsigned short* b_tmp = (unsigned short*)b;

	float tmp = 0;
	float tmp2 = 0;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]);
		tmp2 = sign(tmp);
		b_tmp[id] = __float2half_rn(tmp2);
	}
}

template <typename T,class P>
__global__ void absPowImage_k_half(T *a, T *b, unsigned long int n,P power)
{
	unsigned int xn,yn,zn,id;
	xn=threadIdx.x+blockIdx.x*blockDim.x;
	yn=threadIdx.y+blockIdx.y*blockDim.y;
	zn=threadIdx.z+blockIdx.z*blockDim.z;

	unsigned short* b_tmp = (unsigned short*)b;

	P power_r = power;

	id=xn+yn*xVolumePixelNb_GPU+zn*xVolumePixelNb_GPU*yVolumePixelNb_GPU;

	float tmp = 0;
	float tmp2 = 0;

	// Make sure we do not go out of bounds
	if (id < n)
	{
		tmp = __half2float(a[id]);
		tmp2 = pow((P)fabs(tmp),power_r);
		b_tmp[id] = __float2half_rn(tmp2);
	}
}


#endif