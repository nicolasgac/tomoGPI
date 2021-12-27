/*
 * Iter3D_algo_TV.cu
 *
  */

#include "Iter3D.cuh"
#include "Iter3D_CPU.cuh"
#include "Iter3D_GPU.cuh"
//#include "Iter3D.MGPUcuh"

/* Total variation*/
template<template<typename> class P, template<typename> class BP, template<typename> class R_Huber, template<typename> class R_GG,template<typename> class C, template<typename> class V, template<typename> class S,typename T>
double* Iter3D<P,BP,R_Huber,R_GG,C,V,S,T>::doPrimalDualFrankWolfeTV(V<T>* volume,S<T>* realSinogram, S<T>* dual_proj, V<T>* dual_vol, T* v_noise,double lambda, double norm_H_grad,int stationnary, unsigned int numit){

	std::cout << "PDFW-TV with unmatched RD/VD Pair"  << std::endl;

	unsigned int it;
	double crit;
	double* crit_vect=(double*) malloc(numit*sizeof(double));
	// steps in primal and dual spaces
	double tau, sigma, alpha;
	T factor_var;
	// int stationnary : about the noise
	// 1 : one variance per pixel, and same variance at each projection angle
	// 2 : same variance for all pixels, at all projection angles
	// 0 or other : one variance per pixel and projection angle (default)
	/*********************************************************************
	Reference : Greg Ongie, Naveen Murthy, Laura Balzano, Jeffrey A. Fessler,
    "A Memory-Efficient Algorithm for Large-Scale Sparsity Regularized Image
	Reconstruction"; The 5th International Conference on Image Formation in
	X-Ray Computed Tomography, 2018
	 **********************************************************************/
	// Over-relaxation step is fixed to 0
	// lambda : regularization parameter
	// norm_H_grad : operator norm of the concatened matrix [sqrt(diag(v_noise))*H;nabla] (computed by power method)
	// use unmatched ray-driven/voxel-driven pair of projector/backprojector

	// volume
	this->setVolume(volume);
	// projections
	S<T>* estimatedSinogram=this->create_sinogram3D();//new S<T>(this->getDetector()->getUDetectorPixelNb(), this->getDetector()->getVDetectorPixelNb(), this->getAcquisition()->getProjectionNb());//,this->cudaArchitectureSino);
	estimatedSinogram->setSinogram(0);
	// gradient of the volume
	V<T>* grad_vol=this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb());//,this->cudaArchitectureVolume);
	grad_vol->setVolume(0);
	// auxiliary volume
	V<T>* aux_vol=this->create_volume();//new V<T>(this->getVolume()->getXVolumeSize(),this->getVolume()->getYVolumeSize(),this->getVolume()->getZVolumeSize(),this->getVolume()->getXVolumePixelNb(),this->getVolume()->getYVolumePixelNb(),this->getVolume()->getZVolumePixelNb(),this->cudaArchitectureVolume);
	aux_vol->setVolume(0);

	// apply Primal-Dual Frank-Wolfe algorithm for TV
	for(it=0;it<numit;it++){

		//steps
		tau=2/(2+it+1);
		sigma=1/(norm_H_grad*norm_H_grad*tau);
		alpha=pow(tau,0.49);

		// estimated projections
		this->getProjector()->doProjection(estimatedSinogram,volume);
		estimatedSinogram->diffSinogram(estimatedSinogram,realSinogram);// difference : H*f-g
		crit=0.5*(estimatedSinogram->sumSinogramWeightedL2(v_noise,stationnary));// criterion
		estimatedSinogram->weightByVariancesNoise(v_noise,stationnary);// weights

		// dual variable having size of projections
		factor_var=T(1/(1+sigma));
		dual_proj->scalarSinogram(factor_var);
		factor_var=T(sigma/(1+sigma));
		estimatedSinogram->scalarSinogram(factor_var);
		dual_proj->addSinogram(estimatedSinogram);

		// dual variable having size of the volume
		factor_var=T(1-alpha);
		dual_vol->scalarVolume(factor_var);

		// gradient in x-direction
		grad_vol->grad_xplus(volume);
		crit=crit+lambda*(grad_vol->getVolumeL1Norm());
		aux_vol->sign_volume(grad_vol);
		grad_vol->grad_xmoins(aux_vol);
		factor_var=T(-alpha*lambda);
		dual_vol->addVolume(grad_vol,factor_var);

		// gradient in y-direction
		grad_vol->grad_yplus(volume);
		crit=crit+lambda*(grad_vol->getVolumeL1Norm());
		aux_vol->sign_volume(grad_vol);
		grad_vol->grad_ymoins(aux_vol);
		factor_var=T(-alpha*lambda);
		dual_vol->addVolume(grad_vol,factor_var);

		// gradient in z-direction
		grad_vol->grad_zplus(volume);
		crit=crit+lambda*(grad_vol->getVolumeL1Norm());
		aux_vol->sign_volume(grad_vol);
		grad_vol->grad_zmoins(aux_vol);
		factor_var=T(-alpha*lambda);
		dual_vol->addVolume(grad_vol,factor_var);

		// backprojection
		this->getBackprojector()->doBackProjection(aux_vol,dual_proj);
		aux_vol->addVolume(dual_vol);
		volume->diffVolume(aux_vol,tau);

		// criterion
		crit_vect[it]=crit;

		std::cout << "\tIteration "<<std::to_string(it+1)<<" of PDFW-TV with RD/VD completed" << std::endl;

	}

	// free memory
	delete estimatedSinogram;
	delete grad_vol;
	delete aux_vol;

	return crit_vect;
}

#include "Iter3D_instances_CPU.cu"
#include "Iter3D_instances_GPU.cu"
//#include "Iter3D_instances_MGPU.cu"