# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 1

CPP      = g++
OBJ      = complex16simd.o qrack.o example.o
LINKOBJ  = complex16simd.o qrack.o example.o
BIN      = example
LIBS     = -lm -lpthread
INCS     =
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -std=c++11 -Wall -pedantic
RM       = rm -f

ifeq (${ENABLE_OPENCL},1)
  LIBS += -lOpenCL
  CXXFLAGS += -DENABLE_OPENCL=1
  QRACKVER = qrack_ocl.cpp
else
  CXXFLAGS += -DENABLE_OPENCL=0
  QRACKVER = qrack.cpp # or optionally 'qrack_serial.cpp'
endif

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	$(RM) $(OBJ)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o $(BIN) $(LIBS)

complex16simd.o: complex16simd.cpp
	$(CPP) -c complex16simd.cpp -o complex16simd.o $(CXXFLAGS)	

qrack.o: $(QRACKVER)
	$(CPP) -c $(QRACKVER) -o qrack.o $(CXXFLAGS)	
