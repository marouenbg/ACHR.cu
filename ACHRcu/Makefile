SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

#------------------------------------------------------------
#
# When you adapt this makefile to compile your CPLEX programs
# please copy this makefile and set CPLEXDIR and CONCERTDIR to
# the directories where CPLEX and CONCERT are installed.
#
#------------------------------------------------------------

#CPLEXDIR      = /opt/ibm/ILOG/CPLEX_Studio1271/cplex
GSLDIR        = /usr/include/gsl
CPLEXDIR      =  $(HOME)/CPLEX_Studio12.7.1/cplex
#CONCERTDIR    = ../../../../concert

# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------


CC  = nvcc -Xptxas="-v" -arch=sm_35 -lineinfo --use_fast_math -rdc=true
#CC  = gcc -O0


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
#-lcudavert



all:
	make all_c

#execute: all
#	make execute_c


CPLEXINCDIR   = $(CPLEXDIR)/include
INCLUDES      = $(GSLDIR)$(EIGENDIR)
GSL_LIBS      = -L/usr/lib
EXDIR         = $(CPLEXDIR)/examples
EXINC         = $(EXDIR)/include
EXDATA        = $(EXDIR)/data
EXSRCC        = $(EXDIR)/src/c
EXSRCC        = ./


CFLAGS  = $(COPT)  -I$(CPLEXINCDIR) -I$(INCLUDES)



#------------------------------------------------------------
#  make all      : to compile the examples. 
#  make execute  : to compile and execute the examples.
#------------------------------------------------------------


C_EX = ACHRCuda



all_c: $(C_EX) $(CX_EX)

clean :
	/bin/rm -rf *.o *~ *.class
	/bin/rm -rf $(C_EX) $(CX_EX) $(CPP_EX)
	/bin/rm -rf *.mps *.ord *.sos *.lp *.sav *.net *.msg *.log *.clp

# ------------------------------------------------------------
#
# The examples
#

ACHRCuda: ACHRCuda.o
	$(CC) $(CFLAGS) $(CLNDIRS) $(GSL_LIBS) -o ACHRCuda ACHRCuda.o $(CLNFLAGS)  
ACHRCuda.o: $(EXSRCC)/ACHRCuda.cu
	#$(CC) -c $(CFLAGS) /home/users/mbenguebila/veryfastFVA/veryfastFVA.c -o veryfastFVA.o -lrt
	#$(CC) -c $(CFLAGS) /home/marouen.benguebila/P:/Projects/veryfastFVA/veryfastFVA.c -o veryfastFVA.o 
	$(CC) -c $(CFLAGS) ACHRCuda.cu -o ACHRCuda.o


# Local Variables:
# mode: makefile
# End:
