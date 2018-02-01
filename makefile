QRACKVER = qrack_serial.cpp #or qrack.cpp or qrack_serial.cpp
CPP      = g++
OBJ      = complex16simd.o qrack.o example.o
LINKOBJ  = complex16simd.o qrack.o example.o
BIN      = example
LIBS     = -lm -lpthread -lOpenCL
INCS     =
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -std=c++11 -pedantic
RM       = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o $(BIN) $(LIBS)

complex16simd.o: complex16simd.cpp
	$(CPP) -c complex16simd.cpp -o complex16simd.o $(CXXFLAGS)	

qrack.o: $(QRACKVER)
	$(CPP) -c $(QRACKVER) -o qrack.o $(CXXFLAGS)	
