TomoBayes_exe 1024  --no_save --ngpu=$1 --nstreams=1  --storage=mem_CPU
TomoBayes_exe 1024  --no_save --ngpu=$1 --nstreams=4  --storage=mem_CPU
TomoBayes_exe 1024  --no_save --ngpu=$1 --nstreams=4  --storage=mem_MGPU
