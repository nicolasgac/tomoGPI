
sudo update-alternatives --remove-all gcc 
sudo update-alternatives --remove-all g++

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$1 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$2 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$1 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$2 20

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
