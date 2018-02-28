# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 0

CPP      = g++
XXD      = xxd
OBJ      = complex16simd.o example.o qrack_base.o par_for.o
BIN      = example
LIBS     = -lm -lpthread
INCS     =
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -std=c++11 -Wall -pedantic
RM       = rm -f

ifeq (${ENABLE_OPENCL},1)
  LIBS += -lOpenCL
  CXXFLAGS += -DENABLE_OPENCL=1
  OBJ += qrack_ocl.o qrack.o
else
  CXXFLAGS += -DENABLE_OPENCL=0
  OBJ += qrack.o
endif

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	$(RM) $(OBJ) qrackcl.hpp

$(BIN): $(OBJ)
	$(CPP) $(OBJ) -o $(BIN) $(LIBS)

ifeq (${ENABLE_OPENCL},1)
qrackcl.hpp: qrack.cl
	${XXD} -i qrack.cl > qrackcl.hpp

qrack_ocl.o: qrackcl.hpp
endif
