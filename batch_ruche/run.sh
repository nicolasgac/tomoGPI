source $HOME/.bash_profile
module purge
module load cuda/10.2.89/intel-19.0.3.199

#!nvidia-smi



#!deviceQuery
#!cd /workdir/gacn/data_tomo_gpi/data3D_0256/phantom3D_0006_shepp/
#!TomoBayes_exe 256


cd /workdir/gacn/data_tomo_gpi/data3D_1024/phantom3D_0101_shepp/
#TomoBayes_exe 1024 --no_save --storage=mem_GPU --ngpu=1
source $HOME/tomo_gpi/TomoBayes/batch_ruche/TomoBayes_ngpu.sh $1
#source $HOME/tomo_gpi/TomoBayes/batch_slurm/script/RTCSA_MGPU.sh

