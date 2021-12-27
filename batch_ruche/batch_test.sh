#! /usr/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=Test
#SBATCH --output=output_test.log
#SBATCH --error=error_test.log
#SBATCH --mail-user=nicolas.gac@l2s.centralesupelec.fr
##SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=16G
##SBATCH --nodelist=ruche-gpu02
##SBATCH --time=30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1



module purge
module load cuda/10.2.89/intel-19.0.3.199

nvidia-smi
/workdir/gacn/samples/bin/x86_64/linux/release/deviceQuery
