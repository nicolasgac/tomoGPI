/*
 * Regularizer.cu
 *
 *      Author: gac
 */


#include "Regularizer.cuh"

template <template<typename> class V,typename T>
Regularizer<V,T>::Regularizer(){}

template <template<typename> class V,typename T>
Regularizer<V,T>::~Regularizer(){}

template <typename V>
Regularizer_half<V>::Regularizer_half(){}

template <typename V>
Regularizer_half<V>::~Regularizer_half(){}

#include "Regularizer_instances_CPU.cu"
#include "Regularizer_instances_GPU.cu"
//#include "Regularizer_instances_MGPU.cu"