/*
 * Regularizer_CPU.cpp
 *
 *      Author: gac
 */


#include "Regularizer_CPU.cuh"

template <typename T>
Regularizer_CPU<T>::Regularizer_CPU(){}

template <typename T>
Regularizer_CPU<T>::~Regularizer_CPU(){}

#include "Regularizer_instances_CPU.cu"