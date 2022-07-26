#sbatch --export=ngpu=1 batch_run.sh 
#sbatch --export=ngpu=2 batch_run.sh
sbatch --export=NGPU=4 --export=SCRIPT='/gpfs/users/gacn/tomo_gpi/TomoBayes/batch_ruche/run.sh' batch_run.sh
#sbatch --export=SCRIPT='/gpfs/users/gacn/tomo_gpi/TomoBayes/batch_slurm/script/astra_moulon.sh' batch_moulon_astra.sh
