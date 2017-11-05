#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ilcplex/cplex.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utilities.cuh"
#include <cublas_v2.h>
#include <ctype.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <math_functions.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
/*#include <iostream>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>*/

#define EPSILON 2.2204e-16 

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

__device__ void correctBounds(double *d_curPoint, double *d_ub, double *d_lb, int nRxns, double *d_prevPoint, double alpha, double beta, double *d_centerPoint){

	for(int i=0;i<nRxns ;i++){
		if(d_curPoint[i]>d_ub[i]){
			d_curPoint[i]=d_ub[i];
		}else if(d_curPoint[i]<d_lb[i]){
			d_curPoint[i]=d_lb[i];
		}
		d_prevPoint[i]=d_curPoint[i];
		d_centerPoint[i]=alpha*d_centerPoint[i]+beta*d_curPoint[i];
	}

}

__device__ void reprojectPoint(double *d_N, int nRxns, int istart, double *d_tmp, double *d_curPoint){

	for(int i=0;i<nRxns-istart;i++){
		d_tmp[i]=0;
		for(int j=0;j<nRxns;j++){
			d_tmp[i]+=d_N[j+i*nRxns]*d_curPoint[j];//here t(N)*Pt
		}
	}
	
	for(int i=0;i<nRxns;i++){
		d_curPoint[i]=0;
		for(int j=0;j<nRxns-istart;j++){
			d_curPoint[i]+=d_N[j*nRxns+i]*d_tmp[j];//here N*tmp
		}
	}
}

__device__ void findMaxAbs(int nRxns, double *d_curPoint, double *d_result, int nMets, double *d_Slin_row, double *dev_max){

	for(int i=0;i<nMets;i++){
		d_result[i]=0;
		for(int j=0;j<nRxns;j++){
			d_result[i]+=d_Slin_row[j+i*nRxns]*d_curPoint[j];
		}
		d_result[i]=abs(d_result[i]);
	}

	double *dev_max_ptr = thrust::max_element(thrust::device,d_result, d_result + nMets);
	dev_max[0] = *dev_max_ptr;
	
}

__device__ void addPoint(int pointCount,double *points,double *d_curPoint, int pointsPerFile, int nRxns){
	for(int i=0;i<nRxns;i++){//in row major format (everything else is column major)
			points[pointCount+pointsPerFile*i]=d_curPoint[i];
		}
}

__device__ void advNextStep(double *d_prevPoint, double *d_curPoint, double *d_u, double d_stepDist, int nRxns){
	for(int i=0;i<nRxns;i++){
		d_curPoint[i]=d_prevPoint[i]+d_stepDist*d_u[i];
	}
}

__device__ void fillrandPoint(double *d_fluxMat,int randpointID, int nRxns, int nPts, double *d_centerPoint, double *d_u,double *d_distUb, double *d_distLb,double  *d_ub,double *d_lb,double *d_prevPoint, int d_nValid,double d_pos, double dTol, double uTol, double d_pos_max, double d_pos_min, double *d_maxStepVec, double *d_minStepVec, double *d_min_ptr, double *d_max_ptr){

	int k;
	double d_norm, d_sum;
	k=0;
	/*square<double>        unary_op;
    	thrust::plus<double> binary_op;
    	double init = 0;*/

	for(int i=0;i<nRxns;i++){
		d_u[i]=d_fluxMat[i+randpointID*nRxns]-d_centerPoint[i];//fluxMAt call is d_randPoint
		d_sum+=pow(d_u[i],2);//maybe save square in unsued vector and sum faster with thrust
	}

	//d_sum = thrust::reduce(thrust::device, d_u2, d_u2+nRxns);//MAYBE JUST increment d_sum instead of thrust
	//d_norm=std::sqrt( thrust::transform_reduce(d_u, d_u+nRxns, unary_op, init, binary_op) );
	d_norm=sqrt(d_sum);
	
	for(int i=0;i<nRxns;i++){
		d_u[i]=d_u[i]/d_norm;
		d_distUb[i]=d_ub[i]-d_prevPoint[i];
		d_distLb[i]=d_prevPoint[i]-d_lb[i];
		if(d_distUb[i]>dTol && d_distLb[i]>dTol){
			if(d_u[i] > uTol){
				d_minStepVec[k]=-d_distLb[i]/d_u[i];
				d_maxStepVec[k]=d_distUb[i]/d_u[i];
				k++;
			}else if(d_u[i] < -uTol){
				d_minStepVec[k]=d_distUb[i]/d_u[i];
				d_maxStepVec[k]=-d_distLb[i]/d_u[i];
				k++;
			}
			d_nValid++;
		}
	}

	double *d_min_ptr_dev = thrust::max_element(thrust::device,d_minStepVec, d_minStepVec + k);
	
	double *d_max_ptr_dev = thrust::min_element(thrust::device,d_maxStepVec, d_maxStepVec + k);

	d_min_ptr[0] = *d_min_ptr_dev;
	d_max_ptr[0] = *d_max_ptr_dev;
}

__device__ void createRandomVec(double *randVector, int stepsPerPoint, curandState_t state){
	for(int i=0;i<stepsPerPoint;i++){
		randVector[i]=(double)curand_uniform(&state);
	}
}

__device__ void createPoint(double *points, int stepCount, int stepsPerPoint, int nWrmup, int nRxns,curandState_t state, double *d_fluxMat, double *d_ub, double *d_lb, double dTol, double uTol, double maxMinTol, int pointsPerFile, int nMets, double *d_Slin_row, double *d_N, int istart, double *d_centerPoint, int totalStepCount, int pointCount, double *d_randVector, double *d_prevPoint, double *d_centerPointTmp){
	
	int randPointId, d_nValid;
	double d_u[1100];
	double d_distUb[1100];
	double d_distLb[1100];
	double d_curPoint[1100];
	//double d_result[1100];becomes d_distUb
	//double d_tmp[1100];becomes d_distLB
	double d_maxStepVec[2200];
	double d_minStepVec[2200];
	double d_pos, d_pos_max, d_pos_min;
	double d_min_ptr[1], d_max_ptr[1];
	double d_stepDist, dev_max[1], alpha, beta;

	d_nValid=0;
	while(stepCount < stepsPerPoint){
		randPointId = ceil(nWrmup*(double)curand_uniform(&state));
		//printf("randPoint id is %d \n",randPointId);
		//randPointId = 9;
		fillrandPoint(d_fluxMat, randPointId, nRxns, nWrmup, d_centerPointTmp, d_u, d_distUb, d_distLb, d_ub, d_lb, d_prevPoint, d_nValid,d_pos, dTol, uTol, d_pos_max, d_pos_min, d_maxStepVec, d_minStepVec, d_min_ptr, d_max_ptr);
		d_stepDist=(d_randVector[stepCount])*(d_max_ptr[0]-d_min_ptr[0])+d_min_ptr[0];
		//d_stepDist=(0.5)*(d_max_ptr[0]-d_min_ptr[0])+d_min_ptr[0];
		//printf("min is %f max is %f step is %f \n",d_min_ptr[0],d_max_ptr[0],d_stepDist);
		if ( ((abs(*d_min_ptr) < maxMinTol) && (abs(*d_max_ptr) < maxMinTol)) || (*d_min_ptr > *d_max_ptr) ){ 
			//nMisses++;
			continue;
		}
		advNextStep(d_prevPoint, d_curPoint, d_u, d_stepDist,nRxns);
		if(totalStepCount % 10 == 0){
			findMaxAbs(nRxns, d_curPoint, d_distUb, nMets, d_Slin_row, dev_max);
			if(*dev_max > 1e-9){
				reprojectPoint(d_N,nRxns,istart,d_distLb,d_curPoint);//possibly do in memory the triple mat multiplication
			}
		}
		
		alpha=(double)(nWrmup+totalStepCount+1)/(nWrmup+totalStepCount+1+1);
		beta=1.0/(nWrmup+totalStepCount+1+1);
		
		correctBounds(d_curPoint, d_ub, d_lb, nRxns, d_prevPoint, alpha, beta, d_centerPointTmp);
		
		stepCount++;
		totalStepCount++;
	}
	addPoint(pointCount, points, d_curPoint, pointsPerFile, nRxns);
	
}

__global__ void stepPointProgress(double *d_Slin_row,int pointsPerFile, double *points, int stepsPerPoint, int nRxns, int nWrmup, double *d_fluxMat, double *d_ub, double *d_lb, double dTol, double uTol, double maxMinTol, int nMets, double *d_N, int istart, double *d_centerPoint){
	int index = blockIdx.x * blockDim.x +threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if(index < pointsPerFile){
		int stepCount, totalStepCount;
		double d_prevPoint[1100], d_centerPointTmp[1100], d_randVector[1100];

		curandState_t state;
		curand_init(clock64(),threadIdx.x,0,&state);

		stepCount=0;
		totalStepCount=0;

		for(int i=0;i<nRxns;i++){
			d_centerPointTmp[i]=d_centerPoint[i];
			d_prevPoint[i]=d_centerPoint[i];
		}

		for(int pointCount=index;pointCount<pointsPerFile;pointCount+=stride){
			createRandomVec(d_randVector, stepsPerPoint, state);
			createPoint(points, stepCount, stepsPerPoint, nWrmup, nRxns, state, d_fluxMat, d_ub, d_lb, dTol, uTol, maxMinTol, pointsPerFile,nMets,d_Slin_row,d_N,istart,d_centerPoint,totalStepCount,pointCount,d_randVector,d_prevPoint,d_centerPointTmp);
		}
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
	cudaFree(d_Slin); //keep d_Slin in memory
	cudaFree(d_Slin_copy);

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
	cudaFree(d_SlinTS);
	cudaFree(d_V);
	cudaFree(d_Vh);
	cudaFree(d_U);
	cudaFree(d_S);
	cusolverDnDestroy(solver_handle);
}

int main(int argc, char **argv){
	double maxMinTol = 1e-9;
	double *h_fluxMat, *h_centerPoint, *d_centerPoint;
	double *cmatval;
	double *h_ub, *h_lb, *d_ub, *d_lb;
	double uTol = 1e-9, *points, *h_points;
	double *h_Slin, *d_Slin,*h_N, *d_N, *d_fluxMat, *d_Slin_row;
	double dTol = 1e-14;
	double elapsedTime;
	struct timespec now, tmstart;
	FILE* stream;
	char *pt;
	CPXENVptr env=NULL;
	CPXLPptr lp=NULL;
	int status, istart=0, row=0;
	int nWrmup=0;
	int nRxns=0,nMets=0, nFiles, pointsPerFile, stepsPerPoint;
	int nzcnt,surplus,surplusbis;
	int *cmatbeg, *cmatind, totalCount;	
	int buffer = 8196*128;
	int nDevices;
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
	for(int j=0;j<nRxns;j++){
		for(int i=cmatbeg[j];i<cmatbeg[j+1];i++){
			h_Slin[j*nMets+cmatind[i]]=cmatval[i];
		}
	}
	
	//Transfer ub and lb and Slin
	gpuErrchk(cudaMemcpy(d_ub,h_ub,nRxns*sizeof(double),cudaMemcpyHostToDevice));	
	gpuErrchk(cudaMemcpy(d_lb,h_lb,nRxns*sizeof(double),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_Slin,h_Slin,nRxns*nMets*sizeof(double),cudaMemcpyHostToDevice));
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
	//computeKernelCuda(h_Slin,nRxns,nMets,&istart,h_N,d_Slin,handle);//Parallel version, based on full SVD, thus require a lot of device memory
	computeKernelSeq(h_Slin,nRxns,nMets,h_N,&istart);//Sequential version,  much faster for models < 10k Rxns, host memory
	gpuErrchk(cudaMalloc(&d_N, (nRxns-istart)*nRxns*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_N,h_N,(nRxns-istart)*nRxns*sizeof(double), cudaMemcpyHostToDevice));
	double alpha=1.0, beta=0.0;

	//declare d_slin_row
	gpuErrchk(cudaMalloc(&d_Slin_row, nRxns*nMets*sizeof(double)));
	cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nRxns, nMets, &alpha, d_Slin, nMets, &beta, NULL, nRxns, d_Slin_row, nRxns));
	gpuErrchk(cudaMemcpy(h_Slin,d_Slin_row,nMets*nRxns*sizeof(double),cudaMemcpyDeviceToHost));

	//Compute center point
	h_centerPoint = (double *)malloc(nRxns*sizeof(double));
	gpuErrchk(cudaMalloc(&d_centerPoint, nRxns*sizeof(double)));
	createCenterPt(h_fluxMat, nWrmup, nRxns, h_centerPoint, handle, d_centerPoint);

	//declare loop variables
	gpuErrchk(cudaMalloc(&d_fluxMat, nRxns*nWrmup*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_fluxMat,h_fluxMat,nRxns*nWrmup*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&points, nRxns*pointsPerFile*sizeof(double)));

        clock_gettime(CLOCK_REALTIME, &now);
        elapsedTime = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
        printf("Null space done in %.5f seconds.\n", elapsedTime);

	//Init total step count
	h_points=(double*)calloc(nRxns*pointsPerFile, sizeof(double));
	int blockSize=64, numBlocks=(pointsPerFile + blockSize - 1)/blockSize;

	//Loop through files
	srand(time(NULL));
	for(int ii=0;ii<nFiles;ii++){
		printf("File %d\n",ii);
		//Initialize points matrix to zero
		cudaMemset(points, 0 , nRxns*pointsPerFile*sizeof(double));
		stepPointProgress<<<numBlocks, blockSize>>>(d_Slin_row,pointsPerFile,points,stepsPerPoint,nRxns,nWrmup,d_fluxMat,d_ub,d_lb,dTol,uTol,maxMinTol,nMets,d_N,istart,d_centerPoint);
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

	//Free cuda memory
	cudaFree(d_centerPoint);
	cublasDestroy(handle);
	cudaFree(d_ub);
	cudaFree(d_lb);
	cudaFree(points);
	cudaFree(d_Slin_row);
	cudaFree(d_fluxMat);
	cudaFree(d_N);
	//cudaFree(d_Slin);

	return 0;
}//main

