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
	CPLEXDIR      = $(HOME)/CPLEX_Studio12.7.1/cplex
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
CLNFLAGS  = -lcplex -lm -lpthread -lrt



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


clean :
	/bin/rm -rf *.o *~ *.class
	/bin/rm -rf $(C_EX) $(CX_EX) $(CPP_EX)
	/bin/rm -rf *.mps *.ord *.sos *.lp *.sav *.net *.msg *.log *.clp

# ------------------------------------------------------------
#
# The examples
#

createWarmupPts: createWarmupPts.o
	$(CC) $(CFLAGS) $(CLNDIRS) -o createWarmupPts createWarmupPts.o $(CLNFLAGS)
createWarmupPts.o: $(EXSRCC)/createWarmupPts.c
	#$(CC) -c $(CFLAGS) /home/users/mbenguebila/veryfastFVA/veryfastFVA.c -o veryfastFVA.o -lrt
	#$(CC) -c $(CFLAGS) /home/marouen.benguebila/P:/Projects/veryfastFVA/veryfastFVA.c -o veryfastFVA.o 
	$(CC) -c $(CFLAGS) createWarmupPts.c -o createWarmupPts.o


# Local Variables:
# mode: makefile
# End:
