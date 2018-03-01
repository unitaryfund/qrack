# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 1

CPP      = g++
XXD      = xxd
OBJ      = complex16simd.o example.o qregister.o par_for.o tests.o
SRC      = $(wildcard *.cpp)
HDRS     = $(wildcard *.hpp)
FORMAT_SRC = ${SRC}
FORMAT_HDRS = $(filter-out catch.hpp, ${HDRS})
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

format:
	clang-format-5.0 -style=file -i ${FORMAT_SRC} ${FORMAT_HDRS}

example.o : tests.hpp
tests.o : tests.hpp
qregister.o : qregister.hpp
qregister_opencl.o : qregister.hpp

$(BIN): $(OBJ)
	$(CPP) $(OBJ) -o $(BIN) $(LIBS)

ifeq (${ENABLE_OPENCL},1)
qregistercl.hpp: qregister.cl
	${XXD} -i qregister.cl > qregistercl.hpp

qregister_opencl.o: qregistercl.hpp oclengine.hpp
tests.o : oclengine.hpp
example.o : oclengine.hpp
endif
