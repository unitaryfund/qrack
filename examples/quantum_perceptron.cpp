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
    const real1 eta = 1.0;

#if ENABLE_OPENCL
    // OpenCL type, if available.
    QInterfacePtr qReg =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPENCL, ControlCount + 1, 0);
#else
    // Non-OpenCL type, if OpenCL is not available.
    QInterfacePtr qReg =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_CPU, ControlCount + 1, 0);
#endif

    bitLenInt inputIndices[ControlCount];
    for (bitLenInt i = 0; i < ControlCount; i++) {
        inputIndices[i] = i;
    }

    QNeuronPtr qPerceptron = std::make_shared<QNeuron>(qReg, inputIndices, ControlCount, ControlCount);

    // Train the network to recognize powers of 2
    bool isPowerOf2;
    bitCapInt perm;
    std::cout << "Learning..." << std::endl;
    for (perm = 0; perm < ControlPower; perm++) {
        std::cout << "Epoch " << (perm + 1U) << " out of " << ControlPower << std::endl;
        qReg->SetPermutation(perm);
        isPowerOf2 = ((perm != 0) && ((perm & (perm - 1U)) == 0));
        qPerceptron->Learn(isPowerOf2, eta);
    }

    for (perm = 0; perm < ControlPower; perm++) {
        qReg->SetPermutation(perm);
        std::cout << "Permutation: " << (int)perm << ", Probability: " << qPerceptron->Predict() << std::endl;
    }


    // Now, we prepare a superposition of all available powers of 2, to predict.
    bitLenInt* powersOf2 = new bitLenInt[ControlCount];
    for (bitLenInt i = 0; i < ControlCount; i++) {
        powersOf2[i] = 1U << i;
    }

#if ENABLE_OPENCL
    // OpenCL type, if available.
    QInterfacePtr qReg2 =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPENCL, ControlLog, 0);
#else
    // Non-OpenCL type, if OpenCL is not available.
    QInterfacePtr qReg2 =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_CPU, ControlLog, 0);
#endif

    qReg->Compose(qReg2);
    qReg->SetPermutation(0);
    qReg->H(ControlCount + 1, ControlLog);
    qReg->IndexedLDA(ControlCount + 1, ControlLog, 0, ControlCount, powersOf2);
    qReg->H(ControlCount + 1, ControlLog);
    qReg->Dispose(ControlCount + 1, ControlLog);

    std::cout << "(Superposition of all powers of 2) Probability: " << qPerceptron->Predict() << std::endl;
}
