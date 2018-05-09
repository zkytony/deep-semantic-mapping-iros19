# Include flags
INC=-Iinclude
# Library path flags
LIBS=-lgmpxx -lgmp
# Location of libDAI library
LIB=./lib
# Compiler
CC=g++
# Compiler flags
CCFLAGS=-Wno-deprecated -Wall -W -Wextra -fpic -O3 -static $(INC)

all : fg_topo

fg_topo : fg_topo.cpp $(LIB)/libdai.a
	$(CC) $(CCFLAGS) -o$@ $< $(LIB)/libdai.a $(LIBS)


# CLEAN
########

.PHONY : clean
clean :
	-rm uai2010-aie-solver
