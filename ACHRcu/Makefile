SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

#------------------------------------------------------------
#
# When you adapt this makefile to compile your CPLEX programs
# please copy this makefile and set CPLEXDIR and CONCERTDIR to
# the directories where CPLEX and CONCERT are installed.
#
#------------------------------------------------------------

GSLDIR        ?= /usr
GLPKDIR       ?= /usr

#Please check the CPLEX directory, the default installation directory is commented below
CPLEXDIR      =  $(HOME)/cplex/cplex
#CPLEXDIR      = /opt/ibm/ILOG/CPLEX_Studio1271/cplex

# ---------------------------------------------------------------------
# Compiler selection (auto-detect CUDA path)
# ---------------------------------------------------------------------

CUDA_PATH     ?= $(shell ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1 | sed 's|/bin/nvcc||')
ifeq ($(CUDA_PATH),)
  CUDA_PATH   = /usr/local/cuda
endif
CC  = $(CUDA_PATH)/bin/nvcc -Xptxas="-v" -arch=sm_37 -lineinfo --use_fast_math -rdc=true


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
CCLNFLAGS = -lm -lpthread -lconcert -lilocplex -lcplex
CLNFLAGS  = -lm -lpthread -lrt -lcplex -lcusolver -lcublas Utilities.cu -lcurand -lgsl -lgslcblas -lcudadevrt

# GLPK link flags
GLPK_CFLAGS   = -I$(GLPKDIR)/include
GLPK_LDFLAGS  = -L$(GLPKDIR)/lib
GLPK_CLNFLAGS = -lm -lpthread -lrt -lglpk -lcusolver -lcublas Utilities.cu -lcurand -lgsl -lgslcblas -lcudadevrt



all:
	make all_c


CPLEXINCDIR   = $(CPLEXDIR)/include
INCLUDES      = $(GSLDIR)
GSL_LIBS      = -L$(GSLDIR)/lib
EXSRCC        = ./


CFLAGS  = $(COPT)  -I$(CPLEXINCDIR) -I$(GSLDIR)/include
CFLAGS_GLPK = $(COPT) $(GLPK_CFLAGS) -I$(GSLDIR)/include -DUSE_GLPK


#------------------------------------------------------------
#  make all      : to compile with CPLEX (default)
#  make glpk     : to compile with GLPK (open-source)
#------------------------------------------------------------


C_EX = ACHRCuda



all_c: $(C_EX) $(CX_EX)

# GLPK target (open-source, no CPLEX needed)
glpk: ACHRCuda_glpk

clean :
	/bin/rm -rf *.o *~ *.class
	/bin/rm -rf $(C_EX) ACHRCuda_glpk $(CX_EX) $(CPP_EX)
	/bin/rm -rf *.mps *.ord *.sos *.lp *.sav *.net *.msg *.log *.clp

# ------------------------------------------------------------
# CPLEX build
# ------------------------------------------------------------

ACHRCuda: ACHRCuda.o
	$(CC) $(CFLAGS) $(CLNDIRS) $(GSL_LIBS) -o ACHRCuda ACHRCuda.o $(CLNFLAGS)  
ACHRCuda.o: $(EXSRCC)/ACHRCuda.cu
	$(CC) -c $(CFLAGS) ACHRCuda.cu -o ACHRCuda.o

# ------------------------------------------------------------
# GLPK build (open-source alternative)
# ------------------------------------------------------------

ACHRCuda_glpk: ACHRCuda_glpk.o
	$(CC) $(CFLAGS_GLPK) $(GLPK_LDFLAGS) $(GSL_LIBS) -o ACHRCuda ACHRCuda_glpk.o $(GLPK_CLNFLAGS)
ACHRCuda_glpk.o: $(EXSRCC)/ACHRCuda.cu
	$(CC) -c $(CFLAGS_GLPK) ACHRCuda.cu -o ACHRCuda_glpk.o

# Local Variables:
# mode: makefile
# End:

test:
	echo "Running test"
	./ACHRCuda ../VFWarmup/lib/P_Putida.mps ../VFWarmup/lib/P_Putida1000warmup.csv 1 100 100
