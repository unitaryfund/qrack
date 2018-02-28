# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 1

CPP      = g++
XXD      = xxd
OBJ      = complex16simd.o example.o qregister.o par_for.o tests.o
BIN      = example
LIBS     = -lm -lpthread
INCS     =
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -std=c++11 -Wall -pedantic
RM       = rm -f

ifeq (${ENABLE_OPENCL},1)
  LIBS += -lOpenCL
  CXXFLAGS += -DENABLE_OPENCL=1
  OBJ += qregister_opencl.o oclengine.o
else
  CXXFLAGS += -DENABLE_OPENCL=0
  OBJ += qregister_software.o
endif

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	$(RM) $(OBJ) qregistercl.hpp

$(BIN): $(OBJ)
	$(CPP) $(OBJ) -o $(BIN) $(LIBS)

ifeq (${ENABLE_OPENCL},1)
qregistercl.hpp: qregister.cl
	${XXD} -i qregister.cl > qregistercl.hpp

qregister_opencl.o: qregistercl.hpp
endif
