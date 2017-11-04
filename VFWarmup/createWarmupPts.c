/* --------------------------------------------------------------------------
 * File: createWarmupPts.c
 * Version 1.0
 * --------------------------------------------------------------------------
 * Licence CC BY 4.0 : Free to share and modify 
 * Author : Marouen BEN GUEBILA - marouen.benguebila@uni.lu
 * --------------------------------------------------------------------------
 */
/* createWarmupPts.c - A hybrid Open MP/MPI parallel optimization of fastFVA
   Usage
      createWarmupPts <datafile>   
      <datafile> : .mps file containing LP problem
 */
/*open mp declaration*/
#include <omp.h>
#include "mpi.h"
 
/* ILOG Cplex declaration*/
#include <ilcplex/cplex.h>
/* Bring in the declarations for the string functions */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
/*Forward declaration*/
static void
   free_and_null     (char **ptr),
   usage             (char *progname);

void copyArrMat(double *x, double **fluxMat, int ind, int n){
	/*Copies an array into a column of a multidimensional array*/
	for(int i=0;i<n;i++){
		fluxMat[i][ind] = x[i];
	}
}

void movePtsBds(CPXENVptr env,CPXLPptr lp, double *x, int n){
	/*Moves solution points within bounds*/
	double *lb = NULL, *ub= NULL;
	
	lb = (double *) malloc (n * sizeof(double));
	ub = (double *) malloc (n * sizeof(double));
	CPXgetlb (env, lp, lb, 0, n-1);
	CPXgetub (env, lp, ub, 0, n-1);
	for(int i=0;i<n;i++){
		if(x[i]<lb[i]){
			x[i] = lb[i];
		}else if (x[i]>ub[i]){
			x[i] = ub[i];
		}		
	}
}

void createCenterPt(double **fluxMat, int nPts, int n, double *centPt){
	/*Creates center point of warmup points*/
	int sum=0;
	
	for(int i=0;i<n;i++){
		sum = 0;
		for(int j=0; j<nPts;j++){
			sum += fluxMat[i][j]; 
		}
		centPt[i] = (double)sum/(nPts);
	}
}

void movePtsCet(double **fluxMat, int nPts, int n, double *centPt){
	/*Normalize warmup points to the center point*/
	for(int i=0;i<n;i++){
		for(int j=0;j<nPts;j++){
			fluxMat[i][j] = fluxMat[i][j]*0.33 + 0.67*centPt[i];
		}
	}
		
}

void fva(CPXLPptr lp, int n, int scaling,double **fluxMat, int rank, int numprocs, int nPts){
	/* The actual Open MP FVA called with CPLEX env, CPLEX LP
	the optimal LP solution and n the number of rows
	*/
	int status;
	int cnt = 1;//number of bounds to be changed
	double zero=0, one=1, mOne=-1;;//optimisation percentage
	int i,j,k,curpreind,tid,nthreads, solstat;
	int chunk = 50, ind, indices[n];
	double *values;//obj function initial array
	double objval, *x = NULL;
	
	/*optimisation loop Max:j=-1 Min:j=+1*/
	#pragma omp parallel private(tid,i,j,status,solstat)
		{
			int iters = 0;
			double wTime = omp_get_wtime();
			tid=omp_get_thread_num();
			if(tid==0){
				nthreads=omp_get_num_threads();
				if(rank==0){
					printf("Number of threads = %d, Number of CPUs = %d\n\n",nthreads,numprocs);
				}
			}
			CPXENVptr     env = NULL;
			CPXLPptr      lpi = NULL;
			env = CPXopenCPLEX (&status);//open cplex instance for every thread
			//status = CPXsetintparam (env, CPX_PARAM_PREIND, CPX_OFF);//deactivate presolving
			lpi = CPXcloneprob(env,lp, &status);//clone problem for every thread
			
			/*set solver parameters*/
			status = CPXsetintparam (env, CPX_PARAM_PARALLELMODE, 1);
			status = CPXsetintparam (env, CPX_PARAM_THREADS, 1);
			status = CPXsetintparam (env, CPX_PARAM_AUXROOTTHREADS, 2);
			
			if (scaling){
				/*Change of scaling parameter*/
				status = CPXsetintparam (env, CPX_PARAM_SCAIND, mOne);//1034 is index scaling parameter
				status = CPXgetintparam (env, CPX_PARAM_SCAIND, &curpreind);
			}
			
			/*Initialize array of objective coefficients*/
			for(k=0;k<n;k++){
				indices[k]=k;
			}
			
			/*Allocate array of zeros*/
			values =(double*)calloc(n, sizeof(double));
			
			/*Allocate solution arrary*/
			x = (double *) malloc (n * sizeof(double));
			
			/*Set seed for every thread*/
			srand((unsigned int)(time(NULL)) ^ omp_get_thread_num());

			#pragma omp for schedule(dynamic,chunk) collapse(2) nowait
				for(j=+1;j>-2;j-=2){
					for(i=rank*nPts/numprocs;i<(rank+1)*nPts/numprocs;i++){
						while(solstat == 0){
							status = CPXchgobj (env, lpi, n, indices, values);//turn all coeffs to zero
							if(i<n){
								status = CPXchgobjsen (env, lpi, j);
								status = CPXchgobj (env, lpi, cnt, &i, &one);//change obj index
								status = CPXlpopt (env, lpi);//solve LP
								status = CPXsolution (env, lpi, &solstat, &objval, x, NULL, NULL, NULL);
							}else{
								for(k=0;k<n;k++){
									values[k]=rand()%1 - 0.5;
								}//create random objective function
								status = CPXchgobjsen (env, lpi, j);//change obj sense
								status = CPXchgobj (env, lpi, n, indices, values);
								status = CPXlpopt (env, lpi);//solve LP
								status = CPXsolution (env, lpi, &solstat, &objval, x, NULL, NULL, NULL);
							}
						}
						iters++;
						
						//adjust results within bounds
						movePtsBds(env, lpi, x, n);
						
						//save results
						if(j==-1){//save results
							ind=2*i;
							copyArrMat(x, fluxMat, ind, n);
						}else{
							ind=2*i+1;
							copyArrMat(x, fluxMat, ind, n);
						}
						
						//reinit solstat
						solstat=0;
					}	
				}	
			
			wTime = omp_get_wtime() - wTime;
			printf("Thread %d/%d of process %d/%d did %d iterations in %f s\n",omp_get_thread_num(),omp_get_num_threads(),rank+1,numprocs,iters,wTime);
		}
}

int main (int argc, char **argv){
	int status = 0;
	double elapsedTime;
	struct timespec now, tmstart;
	double *cost     = NULL;
	double *lb       = NULL;
	double *ub       = NULL;
	double zero=0;
	int cnt=1;
	CPXENVptr     env = NULL;//CPLEX environment
	CPXLPptr      lp = NULL;//LP problem
	int curpreind,i, j,m,n,mOne=-1,scaling=0, nPts;
	double **fluxMat, **globalfluxMat;
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	FILE *fp;
	char fileName[100] = "warmup.csv";
	char modelName[100];
	double *centPt = NULL; // initialize center point
	
	/*Initialize MPI*/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);	
	
	/*Check arg number*/
	if (rank==0){
		if(( argc == 2 ) | ( argc == 3 )){
			printf("\nThe model supplied is %s\n", argv[1]);
			strcpy(modelName,argv[1]);
		}else if( argc > 3 ) {
			printf("Too many arguments supplied.\n");
			goto TERMINATE;
		}else {
			printf("One argument expected.\n");
			goto TERMINATE;
		}
    }
	
	/* Initialize the CPLEX environment */
	env = CPXopenCPLEX (&status);
	if ( env == NULL ) {
		char  errmsg[CPXMESSAGEBUFSIZE];
		fprintf (stderr, "Could not open CPLEX environment.\n");
		CPXgeterrorstring (env, status, errmsg);
		fprintf (stderr, "%s", errmsg);
		goto TERMINATE;
	}
	
	/* Turn off output to the screen */
	status = CPXsetintparam (env, CPXPARAM_ScreenOutput, CPX_OFF);
	if ( status ) {
      fprintf (stderr, 
               "Failure to turn on screen indicator, error %d.\n", status);
      goto TERMINATE;
	}
	
	/* Turn on data checking */
	/*status = CPXsetintparam (env, CPXPARAM_Read_DataCheck, CPX_ON);
	if ( status ) {
		fprintf (stderr, 
				"Failure to turn on data checking, error %d.\n", status);
		goto TERMINATE;
	}*/
	
	/* Create the problem. */
	lp = CPXcreateprob (env, &status, "Problem");
	if ( lp == NULL ) {
		fprintf (stderr, "Failed to create LP.\n");
		goto TERMINATE;
	}
	
	/*Read problem */
	status = CPXreadcopyprob (env, lp, argv[1], NULL);
   
	/*Change problem type*/
	status = CPXchgprobtype(env,lp,CPXPROB_LP);
	
	/*Scaling parameter if coupled model*/
	if ( argc == 3 ) {
		if (atoi(argv[2])==-1){
		/*Change of scaling parameter*/
		scaling = 1;
		status = CPXsetintparam (env, CPX_PARAM_SCAIND, mOne);//1034 is index scaling parameter
		status = CPXgetintparam (env, CPX_PARAM_SCAIND, &curpreind);
		printf("SCAIND parameter is %d\n",curpreind);
		}
	}
	
	/* tic. */
	clock_gettime(CLOCK_REALTIME, &tmstart);
	
	/*Problem size */
	m = CPXgetnumrows (env, lp);
	n = CPXgetnumcols (env, lp);
	
	/*Ask for number of warmup points*/
	if(rank==0){
		printf("How many warmup points should I generate? It should be larger than %d. \n", n*2);
		scanf("%d", &nPts);
		/* Write the output to the screen. */
		printf ("Creating %d warmup points! \n", nPts);
	}
	
	/*Dynamically allocate result vector*/
	globalfluxMat =(double**)calloc(n , sizeof(double*));//dimension of lines
	fluxMat       =(double**)calloc(n , sizeof(double*));
	for(i=0;i<n;i++){//dimension of columns
		fluxMat[i]=(double*)calloc(nPts , sizeof(double));
		globalfluxMat[i]=(double*)calloc(nPts , sizeof(double));
	}
    
	/*Disable dynamic teams*/
	omp_set_dynamic(0); 
	
	/*Allocate space for center point*/
	centPt =(double*)calloc(n, sizeof(double));

	/* Create warmup points */
	fva(lp, n, scaling, fluxMat, rank, numprocs, nPts/2);

	/*Reduce results*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(fluxMat, globalfluxMat, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/*Create center point*/
	createCenterPt(globalfluxMat, nPts, n, centPt);
	
	//for(int k=0;k<n;k++){
	//	printf("%f\n",centPt[k]);
	//}
	
	/*Move points to the center*/
	movePtsCet(globalfluxMat, nPts, n, centPt);
	
	/* Print results*/
	/*if(rank==0){
		for(i=0;i<n;i++){//print results and status 
			printf("Min %d is %.2f status is %.1f \n",i,globalminFlux[i],globalminsolStat[i]);
			printf("Max %d is %.2f status is %.1f \n",i,globalmaxFlux[i],globalmaxsolStat[i]);
		}
	}*/
	
	/*Save to csv file*/
	strcat(modelName, fileName);
	fp=fopen(modelName,"w+");
	if(rank==0){
		for(i=0;i<n;i++){
			for(j=0;j<nPts-1;j++){
				fprintf(fp,"%f,",globalfluxMat[i][j]);
			}
			fprintf(fp,"%f",globalfluxMat[i][nPts-1]);//print last value
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
	
	/*Finalize*/
	clock_gettime(CLOCK_REALTIME, &now);
	elapsedTime = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
	if (rank==0){
		printf("Warmup points created in %.5f seconds.\n", elapsedTime);
	}
	MPI_Finalize();
	
TERMINATE:
   /* Free up the problem as allocated by CPXcreateprob, if necessary */
	if ( lp != NULL ) {
		status = CPXfreeprob (env, &lp);
		if ( status ) {
			fprintf (stderr, "CPXfreeprob failed, error code %d.\n", status);
		}
	}
	
	/* Free up the CPLEX environment, if necessary */
	if ( env != NULL ) {
		status = CPXcloseCPLEX (&env);
		if ( status > 0 ) {
			char  errmsg[CPXMESSAGEBUFSIZE];
			fprintf (stderr, "Could not close CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			fprintf (stderr, "%s", errmsg);
		}
	}
	free_and_null ((char **) &cost);
	free_and_null ((char **) &lb);
	free_and_null ((char **) &ub);
	return (status);
}  /* END main */

/* Function to free up the pointer *ptr, and sets *ptr to NULL */
static void free_and_null (char **ptr){
   if ( *ptr != NULL ) {
      free (*ptr);
      *ptr = NULL;
   }
} /* END free_and_null */  

static void usage (char *progname){
   fprintf (stderr,"Usage: %s -X <datafile>\n", progname);
   fprintf (stderr,"   where X is one of the following options: \n");
   fprintf (stderr,"      r          generate problem by row\n");
   fprintf (stderr,"      c          generate problem by column\n");
   fprintf (stderr," Exiting...\n");
} /* END usage */
