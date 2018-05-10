//STD
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ilcplex/cplex.h>
//CUDA
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utilities.cuh"
#include <cublas_v2.h>
#include <ctype.h>
#include <curand.h>
#include <curand_kernel.h>
//Thrust
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <math_functions.h>
//GSL
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
#include <iostream>
#include <cmath>

#define EPSILON 2.2204e-16 

void computeKernelQRSeq(int nRxns, int nMets, double *S, double *h_N){
	gsl_matrix * A      = gsl_matrix_alloc(nMets,nRxns);

	//copy S in gsl format
	for(int j=0;j<nRxns;j++){
		for(int i=0;i<nMets;i++){
			gsl_matrix_set(A,i,j,S[i+j*nMets]);
		}
	}
	//declare SVD variables
	gsl_matrix * AtA  = gsl_matrix_alloc(nRxns,nRxns); 
	gsl_matrix * Q  = gsl_matrix_alloc(nRxns,nRxns); 
	gsl_matrix * R  = gsl_matrix_alloc(nRxns,nRxns); 
	gsl_vector * work = gsl_vector_alloc(nRxns);
	gsl_vector * tau = gsl_vector_alloc(nRxns);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,1.0, A, A,0.0, AtA);

	//QR of AtA
	gsl_linalg_QR_decomp(AtA, tau);
	gsl_linalg_QR_unpack (AtA, tau, Q, R);

	int k=0;
	for(int j=0;j<nRxns;j++){
		for(int i=0;i<nRxns;i++){
			h_N[k] = gsl_matrix_get(Q,i,j);
			k++;
		}
	}

}

void computeKernelQRCuda(int nRxns, int nMets, double *d_Slin, cublasHandle_t handle, double *h_N){
	int work_size=0;
	int *devInfo;
	double *work;
	double *d_SlinTS, *d_Slin_copy;
	double alpha =1.0, beta=0.0, *d_TAU;
	gpuErrchk(cudaMalloc(&d_Slin_copy, nRxns*nMets*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_Slin_copy,d_Slin,nRxns*nMets*sizeof(double),cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMalloc(&d_SlinTS, nRxns*nRxns*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_TAU, nRxns*sizeof(double)));

        //Compute ST*S
	cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nRxns, nRxns, nMets, &alpha, d_Slin, nMets, d_Slin_copy, nMets, &beta, d_SlinTS, nRxns));

	//Init cusolver
	gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);
	
	//Init memory
	cusolveSafeCall(cusolverDnDgeqrf_bufferSize(solver_handle, nRxns, nRxns, d_SlinTS,nRxns,&work_size));
	gpuErrchk(cudaMalloc(&work, work_size*sizeof(double)));

	//QR
	cusolveSafeCall(cusolverDnDgeqrf(solver_handle,nRxns,nRxns,d_SlinTS,nRxns,d_TAU,work,work_size,devInfo));
	int devInfo_h = 0;	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout	<< "Unsuccessful gerf execution\n\n";

	// --- Initializing the output Q matrix (Of course, this step could be done by a kernel function directly on the device)
	double *h_Q = (double *)malloc(nRxns*nRxns*sizeof(double));
	for(int j = 0; j < nRxns; j++)
		for(int i = 0; i < nRxns; i++)
			if (j == i) h_Q[j + i*nRxns] = 1.;
			else		h_Q[j + i*nRxns] = 0.;

	double *d_Q;			gpuErrchk(cudaMalloc(&d_Q, nRxns*nRxns*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_Q, h_Q, nRxns*nRxns*sizeof(double), cudaMemcpyHostToDevice));

	cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, nRxns, nRxns, nRxns, d_SlinTS, nRxns, d_TAU, d_Q, nRxns, work, work_size, devInfo));
	
	gpuErrchk(cudaMemcpy(h_N, d_Q, nRxns * nRxns * sizeof(double), cudaMemcpyDeviceToHost));

	//free the memory
	cudaFree(d_TAU);
	cudaFree(d_Slin_copy);
	cudaFree(d_Slin);
	cusolverDnDestroy(solver_handle);
}

struct non_negative
{
    __host__ __device__
    bool operator()(const int x)
    {
        return x >= 0;
    }
};

template <typename T>
struct square
{
   __host__  __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

void computeKernelSeq(double *S,int nRxns,int nMets, double *h_N, int *istart){
	/* Sequential null space computation through full SVD with GSL, the SVD is done on AtA because
	GSL requires m>=n. N(AtA)=N(A) is the columns of V corresponding to null singular values
	Possibly a test can be done on the matrix size and the SVD can be done on At, the null 
	space would span the columns of U corresponding to null singular values*/

	//int istart;// index of the first null sv (supposdly sorted - could be tested)
	gsl_matrix * A      = gsl_matrix_alloc(nMets,nRxns);

	//copy S in gsl format
	for(int j=0;j<nRxns;j++){
		for(int i=0;i<nMets;i++){
			gsl_matrix_set(A,i,j,S[i+j*nMets]);
		}
	}
	//declare SVD variables
	gsl_matrix * AtA  = gsl_matrix_alloc(nRxns,nRxns); 
	gsl_matrix * V    = gsl_matrix_alloc(nRxns,nRxns);
	gsl_vector * sv   = gsl_vector_alloc(nRxns);
	gsl_vector * work = gsl_vector_alloc(nRxns);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,1.0, A, A,0.0, AtA);

	//SVD of AtA
	gsl_linalg_SV_decomp(AtA,V,sv,work);
	//find null sv wrt a tol (since they are sorted just find the first one)
	double tol = nRxns *  gsl_vector_max(sv) * EPSILON;
	for(int i=0;i<nRxns;i++){
		if(gsl_vector_get(sv,i) < tol){
			*istart = i;
			break;
		}
	}
	printf("sv_%d = %g\n",*istart,gsl_vector_get(sv,*istart));

	//printf("Kernel dim are %d %d\n",nRxns,nRxns-*istart);
	h_N = (double*)realloc(h_N,nRxns*(nRxns-(*istart))*sizeof(double));//Realloc h_N
	int k=0;
	for(int j=*istart;j<nRxns;j++){
		for(int i=0;i<nRxns;i++){
			h_N[k] = gsl_matrix_get(V,i,j);
			k++;
		}
	}

	//free the memory
	gsl_vector_free(sv);
	gsl_vector_free(work);
	gsl_matrix_free(AtA);
	gsl_matrix_free(V);
}

__device__ void correctBounds(double *d_ub, double *d_lb, int nRxns, double *d_prevPoint, double alpha, double beta, double *d_centerPoint, double *points, int pointsPerFile, int pointCount, int index){

	for(int i=0;i<nRxns ;i++){
		if(points[pointCount+pointsPerFile*i]>d_ub[i]){
			points[pointCount+pointsPerFile*i]=d_ub[i];
		}else if(points[pointCount+pointsPerFile*i]<d_lb[i]){
			points[pointCount+pointsPerFile*i]=d_lb[i];
		}
		d_prevPoint[nRxns*index+i]=points[pointCount+pointsPerFile*i];
		d_centerPoint[nRxns*index+i]=alpha*d_centerPoint[nRxns*index+i]+beta*points[pointCount+pointsPerFile*i];
	}
}

__global__ void reprojectPoint(double *d_N, int nRxns, int istart, double *d_umat, double *points, int pointsPerFile, int pointCount, int index){
	int newindex = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=newindex;i<nRxns-istart;i+=stride){
		d_umat[nRxns*index+i]=0;//d_umat now is d_tmp
		for(int j=0;j<nRxns;j++){
			d_umat[nRxns*index+i]+=d_N[j+i*nRxns]*points[pointCount+pointsPerFile*j];//here t(N)*Pt
		}
	}
}

__global__ void reprojectPoint2(double *d_N, int nRxns, int istart, double *d_umat, double *points, int pointsPerFile, int pointCount,int index){
	int newindex= blockIdx.x * blockDim.x + threadIdx.x;
	int stride= blockDim.x * gridDim.x;

	for(int i=newindex;i<nRxns;i+=stride){
		points[pointCount+pointsPerFile*i]=0;
		for(int j=0;j<nRxns-istart;j++){
			points[pointCount+pointsPerFile*i]+=d_N[j*nRxns+i]*d_umat[nRxns*index+j];//here N*tmp
		}
	}
}

__global__ void findMaxAbs(int nRxns, double *d_umat2, int nMets, int *d_rowVec, int *d_colVec, double *d_val, int nnz, double *points, int pointsPerFile, int pointCount, int index){
	int newindex = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int k=newindex;k<nnz;k+=stride){
		d_umat2[nMets*index+d_rowVec[k]]+=d_val[k]*points[pointCount+pointsPerFile*d_colVec[k]];
	}
	
}

__device__ void advNextStep(double *d_prevPoint, double *d_umat, double d_stepDist, int nRxns, double *points, int pointsPerFile, int pointCount, int index){

	for(int i=0;i<nRxns;i++){
		points[pointCount+pointsPerFile*i]=d_prevPoint[nRxns*index+i]+d_stepDist*d_umat[nRxns*index+i];
	}
}

__device__ void fillrandPoint(double *d_fluxMat,int randpointID, int nRxns, int nPts, double *d_centerPointTmp, double *d_umat,double *d_distUb, double *d_distLb,double  *d_ub,double *d_lb,double *d_prevPoint, double d_pos, double dTol, double uTol, double d_pos_max, double d_pos_min, double *d_maxStepVec, double *d_minStepVec, double *d_min_ptr, double *d_max_ptr, int index){

	int k;
	double d_norm, init;
	k=0;
	init = 0;
	square<double>        unary_op;
    	thrust::plus<double> binary_op;

	for(int i=0;i<nRxns;i++){
		d_umat[nRxns*index+i]=d_fluxMat[i+randpointID*nRxns]-d_centerPointTmp[nRxns*index+i];//fluxMAt call is d_randPoint
	}

	d_norm=std::sqrt( thrust::transform_reduce(thrust::seq,d_umat+(nRxns*index), d_umat+(nRxns*(index+1)), unary_op, init, binary_op) );

	for(int i=0;i<nRxns;i++){
		d_umat[nRxns*index+i]=d_umat[nRxns*index+i]/d_norm;
		d_distUb[nRxns*index+i]=d_ub[i]-d_prevPoint[nRxns*index+i];
		d_distLb[nRxns*index+i]=d_prevPoint[nRxns*index+i]-d_lb[i];
		if(d_distUb[nRxns*index+i]>dTol && d_distLb[nRxns*index+i]>dTol){
			if(d_umat[nRxns*index+i] > uTol){
				d_minStepVec[2*nRxns*index+k]=-d_distLb[nRxns*index+i]/d_umat[nRxns*index+i];
				d_maxStepVec[2*nRxns*index+k]=d_distUb[nRxns*index+i]/d_umat[nRxns*index+i];
				k++;
			}else if(d_umat[nRxns*index+i] < -uTol){
				d_minStepVec[2*nRxns*index+k]=d_distUb[nRxns*index+i]/d_umat[nRxns*index+i];
				d_maxStepVec[2*nRxns*index+k]=-d_distLb[nRxns*index+i]/d_umat[nRxns*index+i];
				k++;
			}
		}
	}

	double *d_min_ptr_dev = thrust::max_element(thrust::seq,d_minStepVec+(2*nRxns*index), d_minStepVec+(2*nRxns*index) + k);
	
	double *d_max_ptr_dev = thrust::min_element(thrust::seq,d_maxStepVec+(2*nRxns*index), d_maxStepVec+(2*nRxns*index) + k);

	d_min_ptr[0] = *d_min_ptr_dev;
	d_max_ptr[0] = *d_max_ptr_dev;
}

__global__ void createRandomVec(double *randVector, int stepsPerPoint, curandState_t state){
        int newindex = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

	for(int i=newindex;i<stepsPerPoint;i+=stride){
		randVector[i]=(double)curand_uniform(&state);
	}
}

__device__ void createPoint(double *points, int stepCount, int stepsPerPoint, int nWrmup, int nRxns, curandState_t state, double *d_fluxMat, double *d_ub, double *d_lb, double dTol, double uTol, double maxMinTol, int pointsPerFile, int nMets, double *d_N, int istart, int totalStepCount, int pointCount, double *d_randVector, double *d_prevPoint, double *d_centerPointTmp, int *d_rowVec, int *d_colVec, double *d_val, int nnz, double *d_umat, int index, double *d_umat2, double *d_distUb, double *d_distLb, double *d_maxStepVec, double *d_minStepVec){

	int randPointId;
	double d_pos, d_pos_max, d_pos_min;
	double d_min_ptr[1], d_max_ptr[1];
	double d_stepDist, alpha, beta, dev_max[1];
	int blockSize=128, blockSize1=128, blockSize2=128;// 64 32 32
	int numBlocks=(nnz + blockSize - 1)/blockSize;
	int numBlocks1=(nRxns-istart + blockSize1 - 1)/blockSize1;
	int numBlocks2=(nRxns + blockSize2 - 1)/blockSize2;
	
	//Init min and max ptr
	d_min_ptr[0]=0;d_max_ptr[0]=0;

	while( ((abs(*d_min_ptr) < maxMinTol) && (abs(*d_max_ptr) < maxMinTol)) || (*d_min_ptr > *d_max_ptr) ){
		randPointId = ceil(nWrmup*(double)curand_uniform(&state));
		//printf("randPoint id is %d \n",randPointId);
		//randPointId = 9;
		fillrandPoint(d_fluxMat, randPointId, nRxns, nWrmup, d_centerPointTmp, d_umat, d_distUb, d_distLb, d_ub, d_lb, d_prevPoint, d_pos, dTol, uTol, d_pos_max, d_pos_min, d_maxStepVec, d_minStepVec, d_min_ptr, d_max_ptr, index);
		d_stepDist=(d_randVector[stepCount])*(d_max_ptr[0]-d_min_ptr[0])+d_min_ptr[0];
		//d_stepDist=(0.5)*(d_max_ptr[0]-d_min_ptr[0])+d_min_ptr[0];		//printf("min is %f max is %f step is %f \n",d_min_ptr[0],d_max_ptr[0],d_stepDist);
			//nMisses++;//Init nMisses to -1
	}

	advNextStep(d_prevPoint, d_umat, d_stepDist, nRxns, points, pointsPerFile, pointCount,index);

	if(totalStepCount % 10 == 0){
		for(int k=0;k<nMets;k++){
               		d_umat2[index*nMets+k]=0;//d_umat is d_result
       		}
		//cudaDeviceSynchronize();
		findMaxAbs<<<numBlocks,blockSize>>>(nRxns, d_umat2, nMets, d_rowVec, d_colVec, d_val, nnz, points, pointsPerFile, pointCount, index);
		//cudaDeviceSynchronize();
	        double *dev_max_ptr = thrust::max_element(thrust::seq,d_umat2 + (nMets*index), d_umat2 + (nMets*(index+1)));
	        dev_max[0] = *dev_max_ptr;
		if(*dev_max > 1e-9){
			cudaDeviceSynchronize();
			//__syncthreads();
		        reprojectPoint<<<numBlocks2,blockSize2>>>(d_N,nRxns,istart,d_umat,points,pointsPerFile,pointCount,index);//possibly do in memory the triple mat multiplication
			cudaDeviceSynchronize();
			//__syncthreads();
			reprojectPoint2<<<numBlocks1,blockSize1>>>(d_N,nRxns,istart,d_umat,points,pointsPerFile,pointCount,index);
			//__syncthreads();
			cudaDeviceSynchronize();
		}
	}
	alpha=(double)(nWrmup+totalStepCount+1)/(nWrmup+totalStepCount+1+1);
	beta=1.0/(nWrmup+totalStepCount+1+1);

	//cudaDeviceSynchronize();
	correctBounds(d_ub, d_lb, nRxns, d_prevPoint, alpha, beta, d_centerPointTmp,points,pointsPerFile,pointCount,index);
}

__global__ void stepPointProgress(int pointsPerFile, double *points, int stepsPerPoint, int nRxns, int nWrmup, double *d_fluxMat, double *d_ub, double *d_lb, double dTol, double uTol, double maxMinTol, int nMets, double *d_N, int istart, int *d_rowVec, int *d_colVec, double *d_val, int nnz, double *d_umat, double *d_umat2, double *d_distUb, double *d_distLb, double *d_maxStepVec, double *d_minStepVec, double *d_prevPoint, double *d_centerPointTmp, double *d_randVector){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int stepCount, totalStepCount, pointCount;

	for(int i=index; i < pointsPerFile*stepsPerPoint; i+=stride){

		totalStepCount=index;
		stepCount =totalStepCount % stepsPerPoint;//Changed modulo by div
		pointCount=totalStepCount / stepsPerPoint;
		index= pointCount;//Always equal to point Count
		
		//printf("totalStepCount is %d stepCount %d pointCount %d index %d \n",totalStepCount,stepCount,pointCount,index);
		curandState_t state;
		curand_init(clock64(),threadIdx.x,0,&state);

		/*createRandomVec(d_randVector, stepsPerPoint, state);*///has to be fixed for every step
		createPoint(points, stepCount, stepsPerPoint, nWrmup, nRxns, state, d_fluxMat, d_ub, d_lb, dTol, uTol, maxMinTol, pointsPerFile,nMets,d_N,istart,totalStepCount,pointCount,d_randVector,d_prevPoint,d_centerPointTmp ,d_rowVec, d_colVec, d_val, nnz, d_umat,index,d_umat2,d_distUb,d_distLb,d_maxStepVec,d_minStepVec);
	}
}

int computenWrmup(char *file, int buffer){
	int iter=0;
	FILE* stream;
	char str[buffer];
	char *pt;
	stream=fopen(file,"r");
	fgets(str,buffer,stream);
	pt= strtok(str,",");
	while(pt != NULL){
		pt=strtok(NULL,",");
		iter++;
	}
	fclose(stream);
	return iter;
}

void parseLine(char *pt, int k,double *h_fluxMat,int nWrmup, int nRxns){
	int iter=0;

	while(pt != NULL){
		double a =atof(pt);
		pt = strtok(NULL,",");
		h_fluxMat[iter*nRxns+k]=a;
		iter++;		
	}
}

void createCenterPt(double *h_fluxMat, int nPts, int nRxns, double *h_centerPoint, cublasHandle_t handle, double *d_centerPoint){
	/*Creates center point of warmup points*/
	double alpha=1.0/nPts,beta=0.0;
	double *h_v,*d_v,*d_fluxMat;

	h_v=(double*)malloc(nPts*sizeof(double));
	for(int i=0;i<nPts;i++){
		h_v[i]=1.0;
	}
	//Allocate device memory
	gpuErrchk(cudaMalloc(&d_v, nPts*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_fluxMat, nPts*nRxns*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_fluxMat,h_fluxMat,nRxns*nPts*sizeof(double),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_v,h_v,nPts*sizeof(double),cudaMemcpyHostToDevice));
	//do the sum
	cublasSafeCall(cublasDgemv(handle,CUBLAS_OP_N,nRxns,nPts,&alpha,d_fluxMat,nRxns,d_v,1,&beta,d_centerPoint,1));
	gpuErrchk(cudaMemcpy(h_centerPoint,d_centerPoint,nRxns*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(d_v);
	cudaFree(d_fluxMat);
}

void computeKernelCuda(double *h_Slin,int nRxns,int nMets, int *istart,double *h_N, double *d_Slin, cublasHandle_t handle){
	/* GPU null space computation through full SVD with CUDA, the SVD is done on AtA because
	CUDA requires m>=n. N(AtA)=N(A) is the columns of V corresponding to null singular values
	Possibly a test can be done on the matrix size and the SVD can be done on At, the null 
	space would span the columns of U corresponding to null singular values*/

	// istart is the index of the first null sv (cuda sorts them)
	int work_size=0;
	int *devInfo, devInfo_h=0;
	double *d_S, *h_S, *work, *h_V, *d_U, *d_Vh, *d_V;
	double *d_SlinTS, *d_Slin_copy;
	double alpha =1.0, beta=0.0, tol;
	gpuErrchk(cudaMalloc(&d_Slin_copy, nRxns*nMets*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_Slin_copy,d_Slin,nRxns*nMets*sizeof(double),cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMalloc(&d_SlinTS, nRxns*nRxns*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_S, nRxns*sizeof(double)));
	h_S = (double*)malloc(nRxns*sizeof(double));
	gpuErrchk(cudaMalloc(&d_U, nRxns*nRxns*sizeof(double)));
	h_V = (double*)malloc(nRxns*nRxns*sizeof(double));
	gpuErrchk(cudaMalloc(&d_Vh, nRxns*nRxns*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_V, nRxns*nRxns*sizeof(double)));

        //Compute ST*S
	cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nRxns, nRxns, nMets, &alpha, d_Slin, nMets, d_Slin_copy, nMets, &beta, d_SlinTS, nRxns));

	//SVD of ST*S because N(S)=N(ST*S) is independant of size (SVD assumes m<n)	
	gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);
	//CUDA SVD init
	cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, nRxns, nRxns, &work_size));
	gpuErrchk(cudaMalloc(&work, work_size*sizeof(double)));
	//CUDA  SVD execution
	cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', nRxns, nRxns, d_SlinTS, nRxns, d_S, d_U, nRxns, d_Vh, nRxns, work, work_size, NULL, devInfo));
	cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, nRxns, nRxns, &alpha, d_Vh, nRxns, &beta, NULL, nRxns, d_V, nRxns));
	gpuErrchk(cudaMemcpy(h_S,d_S,nRxns*sizeof(double),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_V,d_V,nRxns*nRxns*sizeof(double),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	tol = nRxns *  h_S[0] * EPSILON;
	if(devInfo_h!=0)
		printf("SVD Unsuccessful");
	for(int i=0;i<nRxns;i++){
		if(h_S[i]<tol){
			*istart=i;
			printf("h_S[%d] is %.15f \n",i,h_S[i]);
			break;
		}
	}
	int k=0;
	h_N = (double*)realloc(h_N,nRxns*(nRxns-(*istart))*sizeof(double));

	for(int j=(*istart)*nRxns;j<nRxns*nRxns;j++){
		h_N[k]=h_V[j];
		k++;
	}

	//free the memory
	cudaFree(d_Slin); //keep d_Slin in memory
	cudaFree(d_Slin_copy);
	cudaFree(d_SlinTS);
	cudaFree(d_V);
	cudaFree(d_Vh);
	cudaFree(d_U);
	cudaFree(d_S);
	cusolverDnDestroy(solver_handle);
	//Host
	free(h_S);
	free(h_V);
}

__global__ void fillCenterPrev(int nRxns, int pointsPerFile, double *d_centerPoint, double *d_prevPoint, double *d_centerPointTmp, double *d_randVector, int stepsPerPoint){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int blockSize=256, numBlocks=(pointsPerFile*stepsPerPoint + blockSize - 1)/blockSize;

        for(int j=index;j<nRxns;j+=stride){
        	for(int ind=0;ind<pointsPerFile;ind++){
                	d_centerPointTmp[nRxns*ind+j]=d_centerPoint[j];
                        d_prevPoint[nRxns*ind+j]=d_centerPoint[j];//needs to be fixed wtr to prevPoint
                }       
        }       

        curandState_t state;
        curand_init(clock64(),threadIdx.x,0,&state);

        createRandomVec<<<numBlocks,blockSize>>>(d_randVector, stepsPerPoint*pointsPerFile, state);//has to be fixed for eve
}


int main(int argc, char **argv){
	double maxMinTol = 1e-9;
	double *h_fluxMat, *h_centerPoint, *d_centerPoint;
	double *cmatval;
	double *h_ub, *h_lb, *d_ub, *d_lb;
	double uTol = 1e-9, *points, *h_points, *d_umat, *d_umat2;
	double *h_Slin, *d_Slin,*h_N, *d_N, *d_fluxMat, *d_distUb, *d_distLb, *d_randVector;
	double *d_val, *h_val, *d_minStepVec, *d_maxStepVec, *d_prevPoint, *d_centerPointTmp;
	//double *d_Slin_row;
	double dTol = 1e-14;
	double elapsedTime;
	struct timespec now, tmstart;
	FILE* stream;
	char *pt;
	CPXENVptr env=NULL;
	CPXLPptr lp=NULL;
	int status, istart=0, row=0;
	int *h_rowVec, *h_colVec, *d_rowVec, *d_colVec;
	int nWrmup=0;
	int nRxns=0,nMets=0, nFiles, pointsPerFile, stepsPerPoint;
	int nzcnt,surplus,surplusbis;
	int *cmatbeg, *cmatind, totalCount;	
	int buffer = 8196*128;
	int nDevices, nnz;
	char filename[8196];
	/*All matrices are stored in column major format, except points, stored in
	row major format*/

	/*TIC*/
	clock_gettime(CLOCK_REALTIME, &tmstart);

	cudaGetDeviceCount(&nDevices);
	for(int i=0; i < nDevices; i++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop,i);
		printf("Device Number: %d\n",i);
		printf("Device name: %s\n",prop.name);
		printf("Memory clock rate (Khz): %d\n", prop.memoryClockRate);
		printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}

	//Compute total step count
	nFiles=atoi(argv[3]);
	pointsPerFile=atoi(argv[4]);
	stepsPerPoint=atoi(argv[5]);
	totalCount=nFiles*pointsPerFile*stepsPerPoint;
	printf("Total count is %d \n",totalCount); 

	//Read the model
	env = CPXopenCPLEX(&status);
	printf("\nThe model supplied is %s\n", argv[1]);
	lp = CPXcreateprob(env, &status, "Problem");
	CPXreadcopyprob(env, lp, argv[1], NULL);
	CPXchgprobtype(env,lp,CPXPROB_LP);
	nMets = CPXgetnumrows (env, lp);
	nRxns = CPXgetnumcols (env, lp);
	printf("nRxns egale a %d  \n",nRxns);
	printf("nMets egale a %d  \n",nMets);	
	h_ub=(double*)calloc(nRxns, sizeof(double));
	gpuErrchk(cudaMalloc(&d_ub, nRxns*sizeof(double)));
	h_lb=(double*)calloc(nRxns, sizeof(double));
	gpuErrchk(cudaMalloc(&d_lb, nRxns*sizeof(double)));
	cmatbeg=(int*)malloc((unsigned) (nRxns+1)*sizeof(int));
	CPXgetub(env,lp,h_ub,0,nRxns-1);
	CPXgetlb(env,lp,h_lb,0,nRxns-1);
	status = CPXgetcols (env, lp, &nzcnt,cmatbeg,NULL,NULL,0,&surplus,0,nRxns-1);
	printf("the value of surplus is %d \n",surplus);
	if(status != CPXERR_NEGATIVE_SURPLUS){
		if(status!=0){
			printf("CPXgetcols for surplus failed, status =%d\n",status);
		}
		printf("All columns in range[%d, %d] are empty.\n",0,nRxns-1);
	}
	surplus=-surplus;
	cmatbeg[nRxns]=surplus;//Should find a better fix
	cmatind=(int*)malloc((unsigned) (1+surplus)*sizeof(int));
	cmatval=(double*)malloc((unsigned) (1+surplus)*sizeof(double));
	status= CPXgetcols(env,lp,&nzcnt,cmatbeg,cmatind,cmatval,surplus,&surplusbis,0,nRxns-1);
	printf("status is %d (0 means all good) \n",status);
	printf("the value of surplus is %d \n",surplusbis);
	//Initialize S
	h_Slin=(double*)calloc(nMets*nRxns, sizeof(double));
	gpuErrchk(cudaMalloc(&d_Slin, nRxns*nMets*sizeof(double)));
	//populate the S matrix
	nnz=0;
	for(int j=0;j<nRxns;j++){
		for(int i=cmatbeg[j];i<cmatbeg[j+1];i++){
			h_Slin[j*nMets+cmatind[i]]=cmatval[i];
			nnz++;
		}
	}
	printf("nnz is %d \n",nnz);
	//Transform into sparse format (CSR format)
	h_rowVec=(int*)calloc(nnz, sizeof(int));
	h_colVec=(int*)calloc(nnz, sizeof(int));
	h_val=(double*)calloc(nnz, sizeof(double));
	nnz=0;
	for(int i=0;i<nMets;i++){
		for(int j=0;j<nRxns;j++){
			if(h_Slin[i+j*nMets]!=0){
				h_rowVec[nnz]=i;
				h_colVec[nnz]=j;
				h_val[nnz]=h_Slin[i+j*nMets];
				nnz++;
			}
		}
	}
	printf("nnz is %d \n",nnz);
	gpuErrchk(cudaMalloc(&d_rowVec, nnz*sizeof(int)));
	gpuErrchk(cudaMalloc(&d_colVec, nnz*sizeof(int)));
	gpuErrchk(cudaMalloc(&d_val, nnz*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_rowVec,h_rowVec,nnz*sizeof(int),cudaMemcpyHostToDevice));	
	gpuErrchk(cudaMemcpy(d_colVec,h_colVec,nnz*sizeof(int),cudaMemcpyHostToDevice));	
	gpuErrchk(cudaMemcpy(d_val,h_val,nnz*sizeof(double),cudaMemcpyHostToDevice));	

	//Transfer ub and lb and Slin
	gpuErrchk(cudaMemcpy(d_ub,h_ub,nRxns*sizeof(double),cudaMemcpyHostToDevice));	
	gpuErrchk(cudaMemcpy(d_lb,h_lb,nRxns*sizeof(double),cudaMemcpyHostToDevice));
	CPXfreeprob(env,&lp);
	CPXcloseCPLEX(&env);

	//Read number of warmup points
	nWrmup = computenWrmup(argv[2],buffer);
	printf("nWrmup egale a %d \n",nWrmup);
        stream = fopen(argv[2],"r");
	h_fluxMat= (double*)calloc(nWrmup*nRxns, sizeof(double*));//should be init to nRxns

	//Read all points, matrix is in column-major format
	char str[buffer];
	while(fgets(str, buffer, stream)){
		pt=strtok(str,",");
		parseLine(pt,row,h_fluxMat,nWrmup,nRxns);
		row++;
	}
	//Close file
	fclose(stream);

 	//Call cublas kernel
        cublasHandle_t handle;
        cublasCreate(&handle);

	//Find the right null space of the S matrix
	h_N=(double*)malloc(nRxns*nRxns*sizeof(double));//Larger than actual size
	gpuErrchk(cudaMemcpy(d_Slin,h_Slin,nRxns*nMets*sizeof(double),cudaMemcpyHostToDevice)); //Parallel version
	computeKernelCuda(h_Slin,nRxns,nMets,&istart,h_N,d_Slin,handle);//Parallel version, based on full SVD, thus require a lot of device memory
	//computeKernelSeq(h_Slin,nRxns,nMets,h_N,&istart);//Sequential version,  much faster for models < 10k Rxns, host memory
	//istart=0;
	//computeKernelQRCuda(nRxns, nMets, d_Slin, handle, h_N);
	//computeKernelQRSeq(nRxns, nMets, h_Slin, h_N);

	//Copy the matrix
	gpuErrchk(cudaMalloc(&d_N, (nRxns-istart)*nRxns*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_N,h_N,(nRxns-istart)*nRxns*sizeof(double), cudaMemcpyHostToDevice));

	//Compute center point
	h_centerPoint = (double *)malloc(nRxns*sizeof(double));
	gpuErrchk(cudaMalloc(&d_centerPoint, nRxns*sizeof(double)));
	createCenterPt(h_fluxMat, nWrmup, nRxns, h_centerPoint, handle, d_centerPoint);

	//declare loop variables
	gpuErrchk(cudaMalloc(&d_fluxMat, nRxns*nWrmup*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_fluxMat,h_fluxMat,nRxns*nWrmup*sizeof(double), cudaMemcpyHostToDevice));
	//d_umat is column-major format
	int blockSize=256, numBlocks=(pointsPerFile*stepsPerPoint + blockSize - 1)/blockSize;
	int blockSize2=256, numBlocks2=(nRxns + blockSize2 - 1)/blockSize2;
	gpuErrchk(cudaMalloc(&d_umat, nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_umat2, nMets*pointsPerFile*sizeof(double)));//could be removed and replaced by d_umat
	gpuErrchk(cudaMalloc(&d_distUb, nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_distLb, nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_minStepVec, 2*nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_maxStepVec, 2*nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_prevPoint, nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_centerPointTmp, nRxns*pointsPerFile*sizeof(double)));
	gpuErrchk(cudaMalloc(&d_randVector, pointsPerFile*stepsPerPoint*sizeof(double)));
	//could be heavily optimized use one dumat per block, use threadID, put the correct number of threads
	gpuErrchk(cudaMalloc(&points, nRxns*pointsPerFile*sizeof(double)));        //Initialize prevPoint and centerPoint

	//Fill center point and previous point
	fillCenterPrev<<<numBlocks2,blockSize2>>>(nRxns, pointsPerFile,d_centerPoint,d_prevPoint,d_centerPointTmp,d_randVector,stepsPerPoint);

	//declare totalStepCount
	/*int *totalStepCount;
	gpuErrchk(cudaMalloc(&totalStepCount, sizeof(int)));
	cudaMemset(totalStepCount, 0 , sizeof(int));*/

        clock_gettime(CLOCK_REALTIME, &now);
        elapsedTime = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
        printf("Null space done in %.5f seconds.\n", elapsedTime);

	//Init total step count
	h_points=(double*)calloc(nRxns*pointsPerFile, sizeof(double));

	//Loop through files
	srand(time(NULL));
	for(int ii=0;ii<nFiles;ii++){
		printf("File %d\n",ii);
		//Initialize points matrix to zero
		cudaMemset(points, 0 , nRxns*pointsPerFile*sizeof(double));
		cudaDeviceSynchronize();
		stepPointProgress<<<numBlocks,blockSize>>>(pointsPerFile,points,stepsPerPoint,nRxns,nWrmup,d_fluxMat,d_ub,d_lb,dTol,uTol,maxMinTol,nMets,d_N,istart,d_rowVec, d_colVec, d_val, nnz, d_umat, d_umat2,d_distUb,d_distLb,d_maxStepVec,d_minStepVec,d_prevPoint,d_centerPointTmp,d_randVector);
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(h_points,points,nRxns*pointsPerFile*sizeof(double),cudaMemcpyDeviceToHost));
		filename[0]='\0';//Init file name
		char dest[]="File";
		sprintf(filename,"%d",ii);
		strcat(dest,filename);
		//strcat(dest,".csv"); keep this one commented
		FILE * f =fopen(dest,"wb");
		int j=0;
		for(int i=0;i<nRxns;i++){
			for(j=0;j<(pointsPerFile-1);j++){
				fprintf(f,"%f,",h_points[i*pointsPerFile+j]);
			}
			fprintf(f,"%f",h_points[i*pointsPerFile+(pointsPerFile-1)]);
			fprintf(f,"\n");
		}
		fclose(f);
	}
	//printf("number of misses is %d \n",nMisses);
	//Finalize
	
        clock_gettime(CLOCK_REALTIME, &now);
        elapsedTime = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));

        printf("Sampling done in %.5f seconds.\n", elapsedTime);
	
	//Free memory
	free(h_fluxMat);
	free(h_centerPoint);
	free(cmatval);
	free(h_ub);
	free(h_lb);
	free(h_points);
	free(h_Slin);
	free(h_N);
	free(h_colVec);
	free(h_rowVec);
	free(h_val);

	//Free cuda memory
	cudaFree(d_centerPoint);
	cublasDestroy(handle);
	cudaFree(d_ub);
	cudaFree(d_lb);
	cudaFree(points);
	cudaFree(d_umat);
	cudaFree(d_umat2);
	//cudaFree(d_Slin_row);
	cudaFree(d_fluxMat);
	cudaFree(d_N);
	//cudaFree(d_Slin);
	cudaFree(d_colVec);
	cudaFree(d_rowVec);
	cudaFree(d_minStepVec);
	cudaFree(d_maxStepVec);
	cudaFree(d_val);
	return 0;
}//main

