SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

#------------------------------------------------------------
#
# When you adapt this makefile to compile your CPLEX programs
# please copy this makefile and set CPLEXDIR and CONCERTDIR to
# the directories where CPLEX and CONCERT are installed.
#
#------------------------------------------------------------

ifndef TRAVIS
	#This is not the default path of CPLEX, please change it this variable if you have a different
	#installation path
	CPLEXDIR      = $(HOME)/cplex/cplex
	#CPLEXDIR      = /work/projects/cplex/soft/cplex/12.6.3/cplex/
	#CONCERTDIR    = ../../../../concert
else
	CPLEXDIR      = /home/travis/travis/cplex
endif
# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

ifndef TRAVIS 
	CC  = mpicc -O0 -fopenmp -std=c99 -D_POSIX_C_SOURCE=199309L
else
	CC  = mpicc -O0 -fopenmp -std=c99 -D_POSIX_C_SOURCE=199309L -coverage
endif

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CCOPT = -m64 -O -fPIC -fno-strict-aliasing -fexceptions -DNDEBUG -DIL_STD


# ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------

CPLEXBINDIR   = $(CPLEXDIR)/bin/$(BINDIST)
CPLEXJARDIR   = $(CPLEXDIR)/lib/cplex.jar
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)


CCLNDIRS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
CLNDIRS   = -L$(CPLEXLIBDIR)
CCLNFLAGS = -lconcert -lilocplex -lcplex -lm -lpthread
CLNFLAGS  = -lcplex -lm -lpthread -ldl -lrt

# GLPK settings (no OpenMP — GLPK is not thread-safe, use MPI only)
GLPKDIR       ?= /usr
CC_GLPK       = mpicc -O2 -std=c99 -D_POSIX_C_SOURCE=199309L
GLPK_CFLAGS   = -I$(GLPKDIR)/include
GLPK_LDFLAGS  = -L$(GLPKDIR)/lib



all:
	make all_c

#execute: all
#	make execute_c


CPLEXINCDIR   = $(CPLEXDIR)/include

EXDIR         = $(CPLEXDIR)/examples
EXINC         = $(EXDIR)/include
EXDATA        = $(EXDIR)/data
#EXDATA        = /home/marouen.benguebila/P:/Projects/veryfastFVA
EXSRCC        = $(EXDIR)/src/c
EXSRCC        = ./
#EXSRCC        = /home/marouen.benguebila/P:/Projects/veryfastFVA


CFLAGS  = $(COPT)  -I$(CPLEXINCDIR)



#------------------------------------------------------------
#  make all      : to compile the examples. 
#  make execute  : to compile and execute the examples.
#------------------------------------------------------------


C_EX = createWarmupPts



all_c: $(C_EX) $(CX_EX)

# GLPK target (open-source, no CPLEX needed)
glpk: createWarmupPtsGLPK

clean :
	/bin/rm -rf *.o *~ *.class
	/bin/rm -rf $(C_EX) $(CX_EX) $(CPP_EX) createWarmupPtsGLPK
	/bin/rm -rf *.mps *.ord *.sos *.lp *.sav *.net *.msg *.log *.clp

# ------------------------------------------------------------
#
# CPLEX-based warmup point generation
#

createWarmupPts: createWarmupPts.o
	$(CC) $(CFLAGS) $(CLNDIRS) -o createWarmupPts createWarmupPts.o $(CLNFLAGS)
createWarmupPts.o: $(EXSRCC)/createWarmupPts.c
	$(CC) -c $(CFLAGS) createWarmupPts.c -o createWarmupPts.o

# ------------------------------------------------------------
#
# GLPK-based warmup point generation (open-source alternative)
#

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  GLPK_EXTRA_LIBS = -lrt
else
  GLPK_EXTRA_LIBS =
endif

createWarmupPtsGLPK: createWarmupPtsGLPK.o
	$(CC_GLPK) -o createWarmupPtsGLPK createWarmupPtsGLPK.o $(GLPK_LDFLAGS) -lglpk -lm -lpthread $(GLPK_EXTRA_LIBS)
createWarmupPtsGLPK.o: $(EXSRCC)/createWarmupPtsGLPK.c
	$(CC_GLPK) $(GLPK_CFLAGS) -c createWarmupPtsGLPK.c -o createWarmupPtsGLPK.o

test:
	echo "Running CPLEX test"
	mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=2 ./createWarmupPts lib/P_Putida.mps 5000

test-glpk:
	echo "Running GLPK test"
	mpirun -np 1 --bind-to none ./createWarmupPtsGLPK lib/P_Putida.mps 5000
# Local Variables:
# mode: makefile
# End:
