#rm -rf /opt/apps/TOMO_GPI_install/bin
#rm -rf /opt/apps/TOMO_GPI_install/mex
#rm -rf /opt/apps/TOMO_GPI_install/lib
mkdir -p /opt/apps/tomo_gpi/bin
mkdir -p /opt/apps/tomo_gpi/mex
mkdir -p /opt/apps/tomo_gpi/lib
mkdir -p /opt/apps/tomo_gpi/Matlab
mkdir -p /opt/apps/tomo_gpi/script
cp -r Matlab/Tomo8 /opt/apps/tomo_gpi/Matlab/
cp bin/* /opt/apps/tomo_gpi/bin/
cp mex/* /opt/apps/tomo_gpi/mex/
cp lib/* /opt/apps/tomo_gpi/lib/
cp script/* /opt/apps/tomo_gpi/script/

