/* --------------------------------------------------------------------------
 * File: createWarmupPtsGLPK.c
 * Version 2.0
 * --------------------------------------------------------------------------
 * Licence CC BY 4.0 : Free to share and modify
 * Author : Marouen BEN GUEBILA
 * --------------------------------------------------------------------------
 * GLPK-based implementation of warmup point generation for ACHR sampling.
 * Replaces the CPLEX dependency with the open-source GLPK solver.
 * Note: GLPK is not thread-safe, so this version uses MPI-only parallelism
 * (no OpenMP). Use multiple MPI ranks for parallel warmup generation.
 * --------------------------------------------------------------------------
 */
/* createWarmupPtsGLPK.c - MPI parallel warmup point generator using GLPK
   Usage
      createWarmupPtsGLPK <datafile> <nPts> [-1]
      <datafile> : .mps file containing LP problem
      <nPts>     : number of warmup points to generate (must be > 2*nRxns)
      -1         : optional, enable scaling
 */
#include "mpi.h"
#include <glpk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

void copyArrMat(double *x, double **fluxMat, int ind, int n){
	for(int i=0;i<n;i++){
		fluxMat[i][ind] = x[i];
	}
}

void movePtsBds(glp_prob *lp, double *x, int n){
	for(int i=0;i<n;i++){
		double lb = glp_get_col_lb(lp, i+1);
		double ub = glp_get_col_ub(lp, i+1);
		if(x[i] < lb){
			x[i] = lb;
		}else if(x[i] > ub){
			x[i] = ub;
		}
	}
}

void createCenterPt(double **fluxMat, int nPts, int n, double *centPt){
	double sum = 0.0;
	for(int i=0;i<n;i++){
		sum = 0.0;
		for(int j=0; j<nPts; j++){
			sum += fluxMat[i][j];
		}
		centPt[i] = sum / nPts;
	}
}

void movePtsCet(double **fluxMat, int nPts, int n, double *centPt){
	for(int i=0;i<n;i++){
		for(int j=0;j<nPts;j++){
			fluxMat[i][j] = fluxMat[i][j]*0.33 + 0.67*centPt[i];
		}
	}
}

void fva(glp_prob *lp, int n, int scaling, double **fluxMat, int rank, int numprocs, int nPts){
	int i, j, k, solstat;
	int ind;
	double *x = NULL;
	int iters = 0, nfailed = 0;
	struct timespec wstart, wend;

	clock_gettime(CLOCK_REALTIME, &wstart);

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	if(scaling){
		glp_scale_prob(lp, GLP_SF_AUTO);
	}

	/* Solve initial feasibility LP to establish a valid basis.
	   GLPK (unlike CPLEX) needs a valid basis before warm-starting works
	   reliably. Without this, the basis degenerates after ~100-200 solves. */
	glp_set_obj_dir(lp, GLP_MIN);
	for(k = 0; k < n; k++){
		glp_set_obj_coef(lp, k+1, 0.0);
	}
	glp_adv_basis(lp, 0);
	glp_simplex(lp, &parm);
	solstat = glp_get_status(lp);
	if(solstat != GLP_OPT && solstat != GLP_FEAS){
		fprintf(stderr, "Error: initial feasibility LP failed (status=%d)\n", solstat);
	}

	x = (double *)malloc(n * sizeof(double));
	srand((unsigned int)(time(NULL)) ^ rank);

	for(i = rank*nPts/numprocs; i < (rank+1)*nPts/numprocs; i++){
		for(j = +1; j > -2; j -= 2){
			for(k = 0; k < n; k++){
				glp_set_obj_coef(lp, k+1, 0.0);
			}

			if(i < n){
				glp_set_obj_dir(lp, (j == 1) ? GLP_MAX : GLP_MIN);
				glp_set_obj_coef(lp, i+1, 1.0);
			}else{
				for(k = 0; k < n; k++){
					glp_set_obj_coef(lp, k+1, (double)rand()/RAND_MAX - 0.5);
				}
				glp_set_obj_dir(lp, (j == 1) ? GLP_MAX : GLP_MIN);
			}

			/* Strategy 1: primal simplex with warm basis */
			int ret = glp_simplex(lp, &parm);
			solstat = glp_get_status(lp);

			/* Strategy 2: if failed, use Bixby's crash basis (like CPLEX) */
			if(ret != 0 || (solstat != GLP_OPT && solstat != GLP_FEAS)){
				glp_cpx_basis(lp);
				ret = glp_simplex(lp, &parm);
				solstat = glp_get_status(lp);
			}

			/* Strategy 3: try dual simplex with fresh basis */
			if(ret != 0 || (solstat != GLP_OPT && solstat != GLP_FEAS)){
				glp_cpx_basis(lp);
				glp_smcp dparm;
				glp_init_smcp(&dparm);
				dparm.msg_lev = GLP_MSG_OFF;
				dparm.meth = GLP_DUAL;
				ret = glp_simplex(lp, &dparm);
				solstat = glp_get_status(lp);
			}

			/* Strategy 4: for random objectives, try new random objectives */
			if((solstat != GLP_OPT && solstat != GLP_FEAS) && i >= n){
				int retries;
				for(retries = 0; retries < 5; retries++){
					for(k = 0; k < n; k++){
						glp_set_obj_coef(lp, k+1, (double)rand()/RAND_MAX - 0.5);
					}
					glp_cpx_basis(lp);
					ret = glp_simplex(lp, &parm);
					solstat = glp_get_status(lp);
					if(solstat == GLP_OPT || solstat == GLP_FEAS) break;
				}
			}

			if(solstat != GLP_OPT && solstat != GLP_FEAS){
				nfailed++;
				if(nfailed <= 10){
					fprintf(stderr, "Warning: LP solve failed (i=%d, j=%d, status=%d)\n", i, j, solstat);
				}else if(nfailed == 11){
					fprintf(stderr, "Warning: suppressing further LP failure messages...\n");
				}
			}
			iters++;

			for(k = 0; k < n; k++){
				x[k] = glp_get_col_prim(lp, k+1);
			}

			movePtsBds(lp, x, n);

			if(j == -1){
				ind = 2*i;
			}else{
				ind = 2*i+1;
			}
			copyArrMat(x, fluxMat, ind, n);
		}
	}

	clock_gettime(CLOCK_REALTIME, &wend);
	double wTime = (double)((wend.tv_sec+wend.tv_nsec*1e-9) - (double)(wstart.tv_sec+wstart.tv_nsec*1e-9));
	printf("Process %d/%d did %d iterations in %f s\n", rank+1, numprocs, iters, wTime);

	free(x);
}

int main(int argc, char **argv){
	int status = 0;
	double elapsedTime;
	struct timespec now, tmstart;
	int i, j, n, m, scaling = 0, nPts;
	double **fluxMat, **globalfluxMat;
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	FILE *fp;
	char fileName[100] = "warmup.csv";
	char modelName[100], finalName[300];
	double *centPt = NULL;
	glp_prob *lp = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);

	if(rank == 0){
		if(argc < 3 || argc > 4){
			printf("Usage: %s <datafile.mps> <nPts> [-1]\n", argv[0]);
			status = 1;
			goto TERMINATE;
		}
		printf("\nThe model supplied is %s\n", argv[1]);
		strcpy(modelName, argv[1]);
	}

	/* Broadcast model name to all ranks */
	MPI_Bcast(modelName, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	glp_term_out(GLP_OFF);

	lp = glp_create_prob();
	if(glp_read_mps(lp, GLP_MPS_DECK, NULL, argv[1])){
		fprintf(stderr, "Failed to read MPS file.\n");
		goto TERMINATE;
	}

	if(argc == 4){
		if(atoi(argv[3]) == -1){
			scaling = 1;
			glp_scale_prob(lp, GLP_SF_AUTO);
			if(rank == 0) printf("Scaling enabled\n");
		}
	}

	clock_gettime(CLOCK_REALTIME, &tmstart);

	m = glp_get_num_rows(lp);
	n = glp_get_num_cols(lp);

	nPts = atoi(argv[2]);
	if(rank == 0){
		if(nPts < n*2){
			printf("Warning: nPts=%d is less than 2*nRxns=%d, setting nPts=%d\n", nPts, n*2, n*2);
			nPts = n*2;
		}
		printf("Creating %d warmup points! \n", nPts);
	}
	MPI_Bcast(&nPts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	globalfluxMat = (double**)calloc(n, sizeof(double*));
	fluxMat       = (double**)calloc(n, sizeof(double*));
	for(i = 0; i < n; i++){
		fluxMat[i]       = (double*)calloc(nPts, sizeof(double));
		globalfluxMat[i] = (double*)calloc(nPts, sizeof(double));
	}

	centPt = (double*)calloc(n, sizeof(double));

	fva(lp, n, scaling, fluxMat, rank, numprocs, nPts/2);

	MPI_Barrier(MPI_COMM_WORLD);
	for(i = 0; i < n; i++){
		MPI_Allreduce(fluxMat[i], globalfluxMat[i], nPts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}

	createCenterPt(globalfluxMat, nPts, n, centPt);
	movePtsCet(globalfluxMat, nPts, n, centPt);

	modelName[strlen(modelName)-4] = '\0';
	sprintf(finalName, "%s%d%s", modelName, nPts, fileName);
	if(rank == 0){
		fp = fopen(finalName, "w+");
		for(i = 0; i < n; i++){
			for(j = 0; j < nPts-1; j++){
				fprintf(fp, "%f,", globalfluxMat[i][j]);
			}
			fprintf(fp, "%f", globalfluxMat[i][nPts-1]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	clock_gettime(CLOCK_REALTIME, &now);
	elapsedTime = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
	if(rank == 0){
		printf("Warmup points created in %.5f seconds.\n", elapsedTime);
	}
	MPI_Finalize();

TERMINATE:
	if(lp != NULL){
		glp_delete_prob(lp);
	}
	for(i = 0; i < n; i++){
		free(fluxMat[i]);
		free(globalfluxMat[i]);
	}
	free(fluxMat);
	free(globalfluxMat);
	free(centPt);
	return status;
}
