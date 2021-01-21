//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates Grover's search, applied for the purpose of finding a value in a lookup table. (This relies
// on the IndexedLDA/IndexedADC/IndexedSBC methods of Qrack. IndexedADC and IndexedSBC can be shown to be unitary
// operations, while IndexedLDA is unitary up to the requirement that the "value register" is set to zero before
// applying the operation.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iomanip> // For setw
#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

void TagValue(bitCapInt targetPerm, QInterfacePtr qReg, bitLenInt valueStart, bitLenInt valueLength)
{
    // Our "oracle" is true for an input of "100" and false for all other inputs.
    qReg->DEC(targetPerm, valueStart, valueLength);
    qReg->ZeroPhaseFlip(valueStart, valueLength);
    qReg->INC(targetPerm, valueStart, valueLength);
}

int main()
{
    int i;

    // ***Grover's search, to find a value in a lookup table***

    // We search for 100, in the lookup table. All values in lookup table are 1 except a single match.
    // We modify Grover's search, to do this. We use Qrack's IndexedLDA/IndexedADC/IndexedSBC methods, which
    // load/add/substract the key-value pairs of a lookup table of classical memory, into superposition in two quantum
    // registers, an index register and a value register. Measurement of either register should always collapse the
    // state in a random VALID key-value pair from the loaded set. The "oracle" tags the target value part, then we
    // "uncompute" to reach a point where we can flip the phase of the initial state. (See
    // https://en.wikipedia.org/wiki/Amplitude_amplification)

    // At the end, we have the target value with high probability, entangled with the index it was loaded in
    // correspondence with.

    const bitLenInt indexLength = 8;
    const bitLenInt valueLength = 8;
    const bitLenInt carryIndex = indexLength + valueLength;

    // We theoretically know that we're looking for a value part of 100.
    const int TARGET_VALUE = 100;

    // We theoretically don't know what the key is, but for the example only, we define it to prepare and check the
    // result state.
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
        // The "oracle" tags one value permutation, which we know. We don't know the key, yet, but the search will
        // return it.
        TagValue(TARGET_VALUE, qReg, 0, valueLength);

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
