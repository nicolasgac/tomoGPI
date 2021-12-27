#! /usr/bin/bash
module purge
module load gcc/9.2.0/gcc-4.8.5
module load cuda/11.2.0/intel-20.0.2 
#module load cmake/3.16.2/gcc-9.2.0
module load matlab/R2020b/intel-19.0.3.199

export MATLAB="/gpfs/softs/softwares/MATLAB/R2020b"
# Add cuda local executables
#export PATH=$PATH:/usr/local/cuda/bin/
export TOMOGPI_DIR="$HOME/tomo_gpi/TomoGPI"
#export PATH="$PATH:$TOMOBAYES_DIR/build/"

source $HOME/.bash_profile
echo $TOMOGPI_DIR
cd $TOMOGPI_DIR
rm -rf build
mkdir build
cd build
$HOME/cmake-3.22.1-linux-x86_64/bin/cmake ..
make -j4
