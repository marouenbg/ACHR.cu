#!/bin/sh

#1. Install Cuda 8.0 for 64-bit machines
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64

sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64-deb

sudo apt-key add /var/cuda-repo-10-0/7fa2af80.pub

rm cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64-deb

sudo apt-get update

sudo apt-get install aptitude

sudo aptitude install cuda

export PATH=/usr/local/cuda-8.0/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

#2. Install GSL
sudo apt-get install libgsl-dev


