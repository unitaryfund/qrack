# 1 to use OpenCL-based optimizations
ENABLE_OPENCL ?= 0

CPP      = g++
XXD      = xxd
OBJ      = complex16simd.o qregister.o par_for.o qregister_software.o qregister_factory.o separatedunit.o
SRC      = $(wildcard *.cpp)
HDRS     = $(wildcard *.hpp)
FORMAT_SRC = ${SRC} qregister.cl
FORMAT_HDRS = $(filter-out catch.hpp, ${HDRS})

TEST_OBJ = test_main.o tests.o
TEST_BIN = unittests
QRACK_LIB=libqrack.a
LIBS     = -lm -lpthread
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -ggdb -std=c++11 -Wall -Werror -DCATCH_CONFIG_FAST_COMPILE
LDFLAGS  =
RM       = rm -f

OPENCL_AMDSDK = /opt/AMDAPPSDK-3.0

ifeq (${ENABLE_OPENCL},1)
  LIBS += -lOpenCL
  CXXFLAGS += -DENABLE_OPENCL=1
  OBJ += qregister_opencl.o oclengine.o
# Support the AMD SDK OpenCL stack
ifneq ($(wildcard $(OPENCL_AMDSDK)/.),)
  CXXFLAGS += -I$(OPENCL_AMDSDK)/include  -Wno-ignored-attributes -Wno-deprecated-declarations
  LDFLAGS += -L$(OPENCL_AMDSDK)/lib/x86_64
endif
else
  CXXFLAGS += -DENABLE_OPENCL=0
endif

.PHONY: all clean test format

all: $(QRACK_LIB) $(TEST_BIN)

clean:
	$(RM) $(OBJ) qregistercl.hpp $(TEST_BIN) $(TEST_OBJ) $(QRACK_LIB)

format:
	clang-format-5.0 -style=file -i $(FORMAT_SRC) $(FORMAT_HDRS)

test: $(TEST_BIN)
	./$(TEST_BIN)

tests.o : tests.hpp catch.hpp
test_main.o : tests.hpp catch.hpp
qregister.o : qregister.hpp
qregister_opencl.o : qregister.hpp
separatedunit.o : separatedunit.hpp

$(TEST_BIN): $(TEST_OBJ) $(QRACK_LIB)
	$(CPP) $(TEST_OBJ) $(QRACK_LIB) -o $(TEST_BIN) $(LDFLAGS) $(LIBS)

$(QRACK_LIB): $(OBJ)
	ar r $(QRACK_LIB) $(OBJ)

ifeq (${ENABLE_OPENCL},1)
qregistercl.hpp: qregister.cl
	${XXD} -i qregister.cl > qregistercl.hpp

qregister_opencl.o: qregistercl.hpp oclengine.hpp
tests.o : oclengine.hpp catch.hpp
example.o : oclengine.hpp catch.hpp
endif
