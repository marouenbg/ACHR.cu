#! /bin/sh

# Exit on error
set -ev

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

# Environment variables
export CFLAGS="-std=c99"
#export MPICH_CC=$CC
export MPICC=mpicc

case "$os" in
    Linux)
       export PATH=$TRAVIS_ROOT/open-mpi/bin:$PATH
       ;;
esac

# Capture details of build
case "$MPI_IMPL" in
    openmpi)
	cd VFWarmup
	make
	#simple test
	#1 core 2 threads
	mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=2 ./createWarmupPts ./lib/P_Putida.mps -1
        export TMPDIR=/tmp
        ;;
esac
