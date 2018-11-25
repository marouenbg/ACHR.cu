#!/bin/sh

#1. Install Cuda 8.0 for 64-bit machines
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb

sudo apt-get update

sudo apt-get install aptitude

sudo apt-get install cuda

export PATH=/usr/local/cuda-8.0/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

#2. Install GSL
sudo apt-get install libgsl-dev
