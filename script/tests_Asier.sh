#!/bin/bash

#Messurer du temps et l'eam et l'eqm

volume_original=$1
projection_analytique=$2
retroprojection_analytique=$3


#for ((a=1; a <= 10 ; a++))
#do 
#echo "Execution numero $a"

echo "**********************"
echo PROJECTION_VI sur CPU temps
echo "**********************"    
projection3D_VI config3D_microCT4.P.txt|grep -e "time" -e "computing"

echo "**********************"
echo PROJECTION_ER sur CPU temps
echo "**********************"
projection3D_ER config3D_microCT4.P.txt|grep -e "time" -e "computing"

echo "**********************"
echo PROJECTION_JOSEPH sur CPU temps
echo "**********************"
projection3D_Joseph config3D_microCT4.P.txt|grep -e "time" -e "computing"

echo "**********************"
echo PROJECTION_ER sur GPU temps
echo "**********************"
projection3D_ER_GPU config3D_microCT4.P.txt|grep -e "time" -e "computing"

echo "**********************"
echo PROJECTION_JOSEPH sur GPU temps
echo "**********************"
projection3D_Joseph_GPU config3D_microCT4.P.txt|grep -e "time" -e "computing"

echo "**********************"
echo BACKPROJECTION sur CPU temps
echo "**********************"
backprojection3D config3D_microCT4.BP.txt|grep -e "time" -e "computing"

echo "**********************"
echo BACKPROJECTION sur GPU temps
echo "**********************"
backprojection3D_GPU config3D_microCT4.BP.txt|grep -e "time" -e "computing"

#done


#Calcul d'erreur
echo "**********************"
echo "eam_rel et eqm_rel entre P_ER_CPU.s et P_ER_GPU.s"
echo "**********************"
eqm_sinogram3D config3D_microCT4.P.txt P_ER_CPU.s P_ER_GPU.s diff.s|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre P_Joseph_CPU.s et P_Joseph_GPU.s"
echo "**********************"
#eqm_sinogram3D config3D_microCT4.P.txt P_Joseph_CPU.s P_Joseph_GPU.s diff.s|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre P_ER_CPU.s et $projection_analytique"
echo "**********************"
eqm_sinogram3D config3D_microCT4.P.txt P_ER_CPU.s $projection_analytique diff.s|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre P_Joseph_CPU.s et $projection_analytique"
echo "**********************"
eqm_sinogram3D config3D_microCT4.P.txt P_Joseph_CPU.s $projection_analytique diff.s|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre P_VI_CPU.s et $projection_analytique"
echo "**********************"
eqm_sinogram3D config3D_microCT4.P.txt P_VI_CPU.s $projection_analytique diff.s|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre BP_CPU.s et BP_GPU.s"
echo "**********************"
eqm_image3D config3D_microCT4.BP.txt BP_GPU.v BP_CPU.v diff.v|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre BP_CPU.s et $retroprojection_analytique ????"
echo "**********************"
#eqm_image3D config3D_microCT4.BP.txt BP_CPU.v $retroprojection_analytique diff.v|grep -e "eam" -e "eqm"

echo "**********************"
echo "eam_rel et eqm_rel entre BP_CPU.s et $volume_original"
echo "**********************"
eqm_image3D config3D_microCT4.BP.txt BP_CPU.v $volume_original diff.v|grep -e "eam" -e "eqm"
