#include <iomanip> // For setw
#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

// Our subroutine returns true only for an input of 100. We are theoretically blind, to this, until the search is
// finished.
const int TARGET_INPUT = 100;

void Oracle(QInterfacePtr qReg)
{
    // Our "oracle" is true for an input of "TARGET_INPUT" and false for all other inputs.
    qReg->DEC(TARGET_INPUT, 0, 8);
    qReg->ZeroPhaseFlip(0, 8);
    qReg->INC(TARGET_INPUT, 0, 8);
    // This ends the "oracle."
}

int main()
{
// ***Grover's search, to invert a black box function***

// Both CPU and GPU types share the QInterface API.
#if ENABLE_OPENCL
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPENCL, 20, 0);
#else
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, 20, 0);
#endif

    int i;

    // Our input to the subroutine "oracle" is 8 bits.
    qReg->SetPermutation(0);
    qReg->H(0, 8);

    std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        // The "oracle" tags one permutation input, which we theoretically don't know.
        Oracle(qReg);

        qReg->H(0, 8);
        qReg->ZeroPhaseFlip(0, 8);
        qReg->H(0, 8);
        qReg->PhaseFlip();
        std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qReg->ProbAll(TARGET_INPUT) << std::endl;
    }

    qReg->MReg(0, 8);

    std::cout << "After measurement:" << std::endl;
    std::cout << "Chance of match:" << qReg->ProbAll(TARGET_INPUT) << std::endl;
}
