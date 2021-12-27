#!/bin/bash

echo "**********************"
echo PROJECTION 3D ER sur CPU
echo "**********************"    
projection3D_ER config3D_microCT4.P.txt

echo "**********************"
echo RETROPROJECTION 3D sur CPU
echo "**********************"    
backprojection3D config3D_microCT4.BP.txt

echo "**********************"
echo PROJECTION 3D ER sur GPU
echo "**********************"  
projection3D_ER_GPU config3D_microCT4.P.txt

echo "**********************"
echo RETROPROJECTION 3D sur GPU
echo "**********************"  
backprojection3D_GPU config3D_microCT4.BP.txt

echo "**********************"
echo Algorithme Iteratif en C
echo "**********************"  
TOMOX_iter3D config3D_microCT4.Iter.txt