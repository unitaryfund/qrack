//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This example demonstrates a (trivial) example of "quantum neuron" or a "quantum perceptron" with the Qrack::QNeuron
// class. This is a type of "neuron" that can learn and predict in superposition, for general machine learning purposes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"
// "qneuron.hpp" defines the QNeuron class.
#include "qneuron.hpp"

using namespace Qrack;

int main()
{
    const bitLenInt ControlCount = 4;
    const bitCapInt ControlPower = 1U << ControlCount;
    const bitLenInt ControlLog = 2;
    const real1 eta = 0.5;

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, ControlCount + 1, 0);

    bitLenInt inputIndices[ControlCount];
    for (bitLenInt i = 0; i < ControlCount; i++) {
        inputIndices[i] = i;
    }

    QNeuronPtr qPerceptron = std::make_shared<QNeuron>(qReg, inputIndices, ControlCount, ControlCount);

    // Train the network to recognize powers of 2
    bool isPowerOf2;
    bitCapInt perm;
    std::cout << "Learning (to recognize powers of 2)..." << std::endl;
    for (perm = 0; perm < ControlPower; perm++) {
        std::cout << "Epoch " << (uint64_t)(perm + 1U) << " out of " << (uint64_t)ControlPower << std::endl;
        qReg->SetPermutation(perm);
        isPowerOf2 = ((perm != 0) && ((perm & (perm - 1U)) == 0));
        qPerceptron->LearnPermutation(isPowerOf2, eta);
    }

    std::cout << "Should be close to 1 for powers of two, and close to 0 for all else..." << std::endl;
    for (perm = 0; perm < ControlPower; perm++) {
        qReg->SetPermutation(perm);
        std::cout << "Permutation: " << (int)perm << ", Probability: " << qPerceptron->Predict() << std::endl;
    }

    // Now, we prepare a superposition of all available powers of 2, to predict.
    unsigned char* powersOf2 = new unsigned char[ControlCount];
    for (bitLenInt i = 0; i < ControlCount; i++) {
        powersOf2[i] = 1U << i;
    }

    QInterfacePtr qReg2 =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, ControlLog, 0);

    qReg->Compose(qReg2);
    qReg->SetPermutation(1U << (ControlCount + 1));
    qReg->H(ControlCount + 1, ControlLog);
    qReg->IndexedLDA(ControlCount + 1, ControlLog, 0, ControlCount, powersOf2);
    qReg->H(ControlCount + 1, ControlLog);
    qReg->Dispose(ControlCount + 1, ControlLog);

    std::cout << "(Superposition of all powers of 2) Probability: " << qPerceptron->Predict() << std::endl;
}
