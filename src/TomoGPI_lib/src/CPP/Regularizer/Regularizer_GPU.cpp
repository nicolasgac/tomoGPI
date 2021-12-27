/*
 * Regularizer_GPU.cpp
 *
 *      Author: gac
 */


#include "Regularizer_GPU.cuh"

template <typename T>
Regularizer_GPU<T>::Regularizer_GPU(){}

template <typename T>
Regularizer_GPU<T>::~Regularizer_GPU(){}

#include "Regularizer_instances_GPU.cu"