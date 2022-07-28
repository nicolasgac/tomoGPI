#! /usr/bin/bash
#SBATCH --partition=gpu_test
##SBATCH --partition=gpu
#SBATCH --job-name=TomoBayes
#SBATCH --output=output_TomoBayes.log
#SBATCH --error=error_TomoBayes.log
#SBATCH --mail-user=nicolas.gac@l2s.centralesupelec.fr
##SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=32G
##SBATCH --nodelist=ruche-gpu05
##SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
source $SCRIPT $NGPU
