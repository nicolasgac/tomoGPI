#!/bin/sh

if [ -z "$1" ]; then
    echo "usage: $0 version" 1>&2
    exit 1
fi

if [ ! -f "/usr/bin/gcc-$1" ] || [ ! -f "/usr/bin/g++-$1" ]; then
    echo "no such version gcc/g++ installed" 1>&2
    exit 1
fi

sudo update-alternatives --set gcc "/usr/bin/gcc-$1"
sudo update-alternatives --set g++ "/usr/bin/g++-$1"
