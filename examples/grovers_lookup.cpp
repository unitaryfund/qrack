#include <iomanip> // For setw
#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

int main()
{
    int i;

    // ***Grover's search, to find a value in a lookup table***

    // We search for 100, in the lookup table. All values in lookup table are 1 except a single match.

    const bitLenInt indexLength = 8;
    const bitLenInt valueLength = 8;
    const bitLenInt carryIndex = indexLength + valueLength;
    const int TARGET_VALUE = 100;
    const int TARGET_KEY = 230;

    // Both CPU and GPU types share the QInterface API.
#if ENABLE_OPENCL
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPENCL, 20, 0);
#else
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, 20, 0);
#endif

    // This array should actually be allocated aligned for best performance, but this will work. We'll talk about
    // alignment for OpenCL in other examples and tutorials.
    unsigned char* toLoad = new unsigned char[1 << indexLength];
    for (i = 0; i < (1 << indexLength); i++) {
        toLoad[i] = 1;
    }
    toLoad[TARGET_KEY] = TARGET_VALUE;

    // Our input to the subroutine "oracle" is 8 bits.
    qReg->SetPermutation(0);
    qReg->H(valueLength, indexLength);
    qReg->IndexedLDA(valueLength, indexLength, 0, valueLength, toLoad);

    // Twelve iterations maximizes the probablity for 256 searched elements, for example.
    // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
    int optIter = M_PI / (4.0 * asin(1.0 / sqrt(1 << indexLength)));

    for (i = 0; i < optIter; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        qReg->DEC(TARGET_VALUE, 0, valueLength);
        qReg->ZeroPhaseFlip(0, valueLength);
        qReg->INC(TARGET_VALUE, 0, valueLength);
        // This ends the "oracle."
        qReg->X(carryIndex);
        qReg->IndexedSBC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        qReg->X(carryIndex);
        qReg->H(valueLength, indexLength);
        qReg->ZeroPhaseFlip(valueLength, indexLength);
        qReg->H(valueLength, indexLength);
        // qReg->PhaseFlip();
        qReg->IndexedADC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        std::cout << "\t" << std::setw(2) << i
                  << "> chance of match:" << qReg->ProbAll(TARGET_VALUE | (TARGET_KEY << valueLength)) << std::endl;
    }

    qReg->MReg(0, 8);

    std::cout << "After measurement (of value, key, or both):" << std::endl;
    std::cout << "Chance of match:" << qReg->ProbAll(TARGET_VALUE | (TARGET_KEY << valueLength)) << std::endl;

    free(toLoad);
}
