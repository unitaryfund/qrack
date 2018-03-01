# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 1

CPP      = g++
XXD      = xxd
OBJ      = complex16simd.o example.o qregister.o par_for.o tests.o qregister_software.o
SRC      = $(wildcard *.cpp)
HDRS     = $(wildcard *.hpp)
FORMAT_SRC = ${SRC}
FORMAT_HDRS = $(filter-out catch.hpp, ${HDRS})
BIN      = example
LIBS     = -lm -lpthread
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -std=c++11 -Wall -Werror
LDFLAGS  =
RM       = rm -f

OPENCL_AMDSDK = /opt/AMDAPPSDK-3.0

ifeq (${ENABLE_OPENCL},1)
  LIBS += -lOpenCL
  CXXFLAGS += -DENABLE_OPENCL=1
  OBJ += qregister_opencl.o oclengine.o
# Support the AMD SDK OpenCL stack
ifneq ($(wildcard ${OPENCL_AMDSDK}/.),)
  CXXFLAGS += -I${OPENCL_AMDSDK}/include  -Wno-ignored-attributes -Wno-deprecated-declarations
  LDFLAGS += -L${OPENCL_AMDSDK}/lib/x86_64
endif
else
  CXXFLAGS += -DENABLE_OPENCL=0
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
	$(CPP) $(OBJ) -o $(BIN) ${LDFLAGS} $(LIBS)

ifeq (${ENABLE_OPENCL},1)
qregistercl.hpp: qregister.cl
	${XXD} -i qregister.cl > qregistercl.hpp

qregister_opencl.o: qregistercl.hpp oclengine.hpp
tests.o : oclengine.hpp
example.o : oclengine.hpp
endif
